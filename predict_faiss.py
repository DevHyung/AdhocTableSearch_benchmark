"""
 FAISS-based index components for dense retriver
"""

import os
import time
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import logging
import pickle
import subprocess
import re

import faiss
import torch
import numpy as np

from model import QueryTableMatcher
from transformers import BertTokenizer
from table_bert import Table, Column

from typing import List, Tuple, Iterator
import random

logger = logging.getLogger()


class TREC_evaluator(object):
    def __init__(self, qrels_file, result_file, trec_cmd = "../trec_eval"):
        self.rank_path = result_file
        self.qrel_path = qrels_file
        self.trec_cmd = trec_cmd

    def get_ndcgs(self, metric='ndcg_cut', qrel_path=None, rank_path=None, all_queries=False):
        if qrel_path is None:
            qrel_path = self.qrel_path
        if rank_path is None:
            rank_path = self.rank_path

        metrics = ['ndcg_cut_5', 'ndcg_cut_10', 'ndcg_cut_15', 'ndcg_cut_20',
                   #'map_cut_5', 'map_cut_10', 'map_cut_15', 'map_cut_20',
                'map', 'mrr', 'recip_rank']  # 'relstring'
        if all_queries:
            results = subprocess.run([self.trec_cmd, '-c', '-m', metric, '-q', qrel_path, rank_path],
                                     stdout=subprocess.PIPE).stdout.decode('utf-8')
            q_metric_dict = dict()
            for line in results.strip().split("\n"):
                seps = line.split('\t')

                metric = seps[0].strip()
                qid = seps[1].strip()
                if metric not in metrics:
                    continue
                if metric != 'relstring':
                    score = float(seps[2].strip())
                else:
                    score = seps[2].strip()
                if qid not in q_metric_dict:
                    q_metric_dict[qid] = dict()
                q_metric_dict[qid][metric] = score
            return q_metric_dict

        else:
            results = subprocess.run([self.trec_cmd, '-c', '-m', metric, qrel_path, rank_path],
                                     stdout=subprocess.PIPE).stdout.decode('utf-8')

            ndcg_scores = dict()
            for line in results.strip().split("\n"):
                seps = line.split('\t')
                metric = seps[0].strip()
                qid = seps[1].strip()
                if metric not in metrics or qid != 'all':
                    continue
                ndcg_scores[seps[0].strip()] = float(seps[2])
            return ndcg_scores


class DenseIndexer(object):
    def __init__(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        self.index_id_to_db_id = []
        self.index = None

    def index_data(self, vector_files: List[str]):
        start_time = time.time()
        buffer = []
        for i, item in enumerate(iterate_encoded_files(vector_files)):
            db_id, doc_vector = item
            buffer.append((db_id, doc_vector))
            if 0 < self.buffer_size == len(buffer):
                # indexing in batches is beneficial for many faiss index types
                self._index_batch(buffer)
                logger.info('data indexed %d, used_time: %f sec.',
                            len(self.index_id_to_db_id), time.time() - start_time)
                buffer = []
        self._index_batch(buffer)

        indexed_cnt = len(self.index_id_to_db_id)
        logger.info('Total data indexed %d', indexed_cnt)
        logger.info('Data indexing completed.')

    def _index_batch(self, data: List[Tuple[object, np.array]]):
        raise NotImplementedError

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        raise NotImplementedError

    def serialize(self, file: str):
        logger.info('Serializing index to %s', file)

        if os.path.isdir(file):
            index_file = os.path.join(file, "index.dpr")
            meta_file = os.path.join(file, "index_meta.dpr")
        else:
            index_file = file + '.index.dpr'
            meta_file = file + '.index_meta.dpr'

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode='wb') as f:
            pickle.dump(self.index_id_to_db_id, f)

    def deserialize_from(self, file: str):
        logger.info('Loading index from %s', file)

        if os.path.isdir(file):
            index_file = os.path.join(file, "index.dpr")
            meta_file = os.path.join(file, "index_meta.dpr")
        else:
            index_file = file + '.index.dpr'
            meta_file = file + '.index_meta.dpr'

        self.index = faiss.read_index(index_file)
        logger.info('Loaded index of type %s and size %d', type(self.index), self.index.ntotal)

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert len(
            self.index_id_to_db_id) == self.index.ntotal, 'Deserialized index_id_to_db_id should match faiss index size'

    def _update_id_mapping(self, db_ids: List):
        self.index_id_to_db_id.extend(db_ids)

# 이건 Similarity
class DenseFlatIndexer(DenseIndexer):

    def __init__(self, vector_sz: int, buffer_size: int = 50000):
        super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)
        self.index = faiss.IndexFlatIP(vector_sz)

    def _index_batch(self, data: List[Tuple[object, np.array]]):
        db_ids = [t[0] for t in data]
        vectors = [np.reshape(t[1], (1, -1)) for t in data]
        vectors = np.concatenate(vectors, axis=0)
        self._update_id_mapping(db_ids)
        self.index.add(vectors)

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        scores, indexes = self.index.search(query_vectors, top_docs)
        # convert to external ids
        db_ids = [[self.index_id_to_db_id[i] for i in query_top_idxs] for query_top_idxs in indexes]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        return result

# 이건 L2 Distance 기반
class DenseHNSWFlatIndexer(DenseIndexer):
    """
     Efficient index for retrieval. Note: default settings are for hugh accuracy but also high RAM usage
    """
    # b 50000, store 512
    def __init__(self, vector_sz: int, buffer_size: int = 50000, store_n: int = 512
                 , ef_search: int = 128, ef_construction: int = 200):
        super(DenseHNSWFlatIndexer, self).__init__(buffer_size=buffer_size)

        # IndexHNSWFlat supports L2 similarity only
        # so we have to apply DOT -> L2 similairy space conversion with the help of an extra dimension
        index = faiss.IndexHNSWFlat(vector_sz + 1, store_n)
        index.hnsw.efSearch = ef_search
        index.hnsw.efConstruction = ef_construction
        self.index = index
        self.phi = None

    def index_data(self, vector_files: List[str]):
        self._set_phi(vector_files)

        super(DenseHNSWFlatIndexer, self).index_data(vector_files)

    def _set_phi(self, vector_files: List[str]):
        """
        Calculates the max norm from the whole data and assign it to self.phi: necessary to transform IP -> L2 space
        :param vector_files: file names to get passages vectors from
        :return:
        """
        phi = 0
        for i, item in enumerate(iterate_encoded_files(vector_files)):
            id, doc_vector = item
            norms = (doc_vector ** 2).sum()
            phi = max(phi, norms)
        logger.info('HNSWF DotProduct -> L2 space phi={}'.format(phi))
        self.phi = phi

    def _index_batch(self, data: List[Tuple[object, np.array]]):
        # max norm is required before putting all vectors in the index to convert inner product similarity to L2
        if self.phi is None:
            raise RuntimeError('Max norm needs to be calculated from all data at once,'
                               'results will be unpredictable otherwise.'
                               'Run `_set_phi()` before calling this method.')

        db_ids = [t[0] for t in data]
        vectors = [np.reshape(t[1], (1, -1)) for t in data]

        norms = [(doc_vector ** 2).sum() for doc_vector in vectors]
        aux_dims = [np.sqrt(self.phi - norm) for norm in norms]
        hnsw_vectors = [np.hstack((doc_vector, aux_dims[i].reshape(-1, 1))) for i, doc_vector in
                        enumerate(vectors)]
        hnsw_vectors = np.concatenate(hnsw_vectors, axis=0)

        self._update_id_mapping(db_ids)
        self.index.add(hnsw_vectors)

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:

        aux_dim = np.zeros(len(query_vectors), dtype='float32')
        query_nhsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
        #logger.info('query_hnsw_vectors %s', query_nhsw_vectors.shape)
        scores, indexes = self.index.search(query_nhsw_vectors, top_docs)
        # convert to external ids
        db_ids = [[self.index_id_to_db_id[i] for i in query_top_idxs] for query_top_idxs in indexes]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        return result

    def deserialize_from(self, file: str):
        super(DenseHNSWFlatIndexer, self).deserialize_from(file)
        # to trigger warning on subsequent indexing
        self.phi = None


def iterate_encoded_files(vector_files: list) -> Iterator[Tuple[object, np.array]]:
    for i, file in enumerate(vector_files):
        #logger.info('Reading file %s', file)
        #with open(file, "rb") as reader:
            #doc_vectors = pickle.load(reader)
            #for doc in doc_vectors:
                #db_id, doc_vector = doc
                #yield db_id, doc_vector
        doc_vector = torch.load(file)
        yield i, doc_vector

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def faiss_search(args, query_dict, qrel_dict):
    # Model, BERT loader
    query_tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    model = QueryTableMatcher(args)
    model = model.load_from_checkpoint(
        checkpoint_path=args.ckpt_file,
    )
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    # Table Indexing
    data_path = args.data_dir
    _table_encode(data_path=data_path, table_encoder=model)

    table_index_dir = Path(os.path.join(args.data_dir, 'faiss/'))
    table_index_list = os.listdir(table_index_dir)
    input_paths = [str(os.path.join(table_index_dir, file)) for file in table_index_list]
    if args.index == 'l2':
        index = DenseHNSWFlatIndexer(768)
    elif args.index == 'dot':
        index = DenseFlatIndexer(768)
    else:
        raise RuntimeError('Index argument error. ')
    index.index_data(input_paths)

    # Query encoding
    result_dict = {}
    sum_of_search_time = 0


    for qid, query in tqdm(query_dict.items()):
        qid_match_tables_list = qrel_dict[qid]
        query_tokenized = query_tokenizer.encode_plus(query,
                                                      max_length=7,
                                                      padding='max_length',
                                                      truncation=True,
                                                      return_tensors="pt"
                                                      ).to(device)
        #qCLS = model.Qmodel(**query_tokenized)[1].detach().numpy()
        # Norm 추가
        qCLS = model.Tmodel.bert(**query_tokenized)[1]
        #qCLS = model.norm(qCLS).detach().numpy()
        qCLS = model.norm(qCLS).cpu().numpy()



        # knn Search
        time0 = time.time()
        top_ids_and_scores = index.search_knn(qCLS, len(input_paths))
        # 0.023339033126831
        sum_of_search_time += time.time() - time0
        indexes = top_ids_and_scores[0][0]
        scores = top_ids_and_scores[0][1]

        db_ids = [table_index_list[i] for i in indexes]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]

        # search
        for idx, qrel_list in enumerate(qid_match_tables_list):
            for rank_table in result:
                if qrel_list[0].strip() == rank_table[0].replace('.faiss', '').strip():  # Tid Check
                    qid_match_tables_list[idx][-1] = rank_table[-1]  # save score
                    break

        # save
        result_dict[qid] = qid_match_tables_list

    print('index search time: Avg %f sec.', sum_of_search_time / len(query_dict.keys()))

    return result_dict


def infer_column_type_from_row_values(numeric_idx_list, heading, body):
    heading_type_dict = {k : 'text' for k in heading}
    for n_idx in numeric_idx_list:
        heading_type_dict[heading[n_idx]] = 'real'
        for i, rows in enumerate(body):
            try:
                float(rows[n_idx].strip().replace('−','-').replace(',','').replace('–','-'))
            except:
                heading_type_dict[heading[n_idx]] = 'text'
                break
    return heading_type_dict


def _table_encode(data_path = './data/', table_encoder = None):
    table_index_dir = Path(os.path.join(data_path, 'faiss/'))
    if os.path.exists(table_index_dir):
        return
    else:
        table_index_dir.mkdir(exist_ok=True)
        path = Path(data_path + 'test.jsonl')
        html_pattern = re.compile(r'<\w+ [^>]*>([^<]+)</\w+>')
        tag_pattern = re.compile(r'<.*?>')
        link_pattern = re.compile(r'\[.*?\|.*?\]')
        with open(path) as f:
            with torch.no_grad():
                for line in tqdm(f.readlines()):
                    if not line.strip():
                        break
                    # 테이블 Meta data parsing ( qid, tid, query, rel )
                    jsonStr = json.loads(line)
                    # 추가
                    tid = jsonStr['docid'].replace('/','@')# TID보정
                    query = jsonStr['query']
                    qid = jsonStr['qid']
                    rel = jsonStr['rel']

                    if qid not in query_dict:
                        # 추가200423 : add_special_tokens
                        query_tokenized = query_tokenizer.encode_plus(query,
                                                                      max_length=max_query_length,
                                                                      add_special_tokens=True,
                                                                      padding='max_length',
                                                                      truncation=True,
                                                                      return_tensors="pt"
                                                                      )
                        query_dict[qid] = query_tokenized  # BERT **input input_ids, seg_ids, mas_ids

                    # Raw Json parsing ( Detail table information )
                    raw_json = json.loads(jsonStr['table']['raw_json'])

                    textBeforeTable = raw_json['textBeforeTable']  # 추후
                    textAfterTable = raw_json['textAfterTable']  # 추후
                    title = raw_json['pageTitle']
                    caption = re.sub(r'[^a-zA-Z0-9]', ' ', raw_json['title']).strip()  # Caption 역할
                    tableOrientation = raw_json['tableOrientation']  # [HORIZONTAL, VERTICAL]

                    headerPosition = raw_json['headerPosition']  # ['FIRST_ROW', 'MIXED', 'FIRST_COLUMN', 'NONE’]
                    hasHeader = raw_json['hasHeader']  # [true, false]
                    keyColumnIndex = raw_json['keyColumnIndex']
                    headerRowIndex = raw_json['headerRowIndex']  # 0 == 첫줄, -1 == 없음

                    heading = []
                    body = []

                    # 방향은 달라도 데이터 표현은 같이 해줘서 우선은 동일하게 코드구성
                    # TODO: 나중에 하나씩 원본 URL들어가서 확인해볼 부분
                    # hasHeader, headerRowIndex가 있든 없든 0번째 줄이 header역할
                    # TODO: 나중에 Keycolumn을 헤더가 없을때 사용할수있을까?
                    if tableOrientation.strip() == "HORIZONTAL":
                        # Col List -> Table
                        table_data = raw_json['relation']
                        col_cnt = len(table_data)
                        row_cnt = len(table_data[0])

                        for row in range(row_cnt):
                            tmp_row_data = []
                            for col in range(col_cnt):
                                tmp_row_data.append(table_data[col][row])
                            body.append(tmp_row_data)

                        # Header
                        for table_col in table_data:
                            heading.append(table_col[0])

                    elif tableOrientation.strip() == "VERTICAL":
                        # Col List -> Table
                        table_data = raw_json['relation']
                        col_cnt = len(table_data)
                        row_cnt = len(table_data[0])

                        for row in range(row_cnt):
                            tmp_row_data = []
                            for col in range(col_cnt):
                                tmp_row_data.append(table_data[col][row])
                            body.append(tmp_row_data)

                        # Header
                        for table_col in table_data:
                            heading.append(table_col[0])

                    else:
                        print(">>> Check the table data")
                        exit(-1)

                    # Heading preprocessing + link remove
                    heading_str = ' '.join(heading)
                    if html_pattern.search(heading_str):
                        if link_pattern.search(heading_str):  # 같이 있는 경우
                            heading = [re.sub(tag_pattern, '', column).strip() for column in heading]
                            for idx, column in enumerate(heading):
                                if link_pattern.search(column):
                                    real_text = link_pattern.search(column).group().split('|')[-1][:-1].strip()
                                    heading[idx] = real_text
                        else:
                            heading = [re.sub(html_pattern, '', column).strip() for column in heading]

                    # Row preporcessing + link remove
                    cell_sum_str = ''
                    for rows in body:
                        cell_sum_str += ' '.join(rows)

                    if html_pattern.search(cell_sum_str):
                        if link_pattern.search(cell_sum_str):  # 같이 있으면
                            for i, rows in enumerate(body):
                                for j, cell in enumerate(rows):
                                    if link_pattern.search(cell):
                                        cell = re.sub(tag_pattern, '', cell).strip()
                                        real_text = link_pattern.search(cell).group().split('|')[-1][:-1]
                                        body[i][j] = real_text
                                    else:
                                        cell = re.sub(html_pattern, '', cell).strip()
                                        body[i][j] = cell

                        else:
                            row_list = []
                            for rows in body:
                                row_list.append([re.sub(html_pattern, '', row).strip() for row in rows])
                            body = row_list

                    # Infer column type
                    # heading_type_dict = self.infer_column_type_from_row_values(numericIdx, heading, body)

                    # TODO: Context부분을 다양하게 주는부분, 비교실험 해볼부분임
                    # TODO: Special Token을 추가 안해두 되는지 비교실험
                    # TODO: Text Before after부분 활용?
                    caption = " ".join(heading) + " " + title + " " + caption
                    caption_rep = table_encoder.Tmodel.tokenizer.tokenize(caption)

                    column_rep = Table(id=title,
                                       header=[Column(h.strip(), 'text') for h in heading],
                                       data=body
                                       ).tokenize(table_encoder.Tmodel.tokenizer)

                    try:
                        context_encoding, column_encoding, _ = table_encoder.Tmodel.encode(contexts=[caption_rep],
                                                                                           tables=[column_rep])
                    except:
                        print(">>> Except table")
                        continue
                        #
                        column_rep = Table(id=title,
                                           header=[Column(h.strip(), 'text') for h in heading],
                                           data=[['']]
                                           ).tokenize(table_encoder.Tmodel.tokenizer)
                        context_encoding, column_encoding, _ = table_encoder.Tmodel.encode(contexts=[caption_rep],
                                                                                           tables=[column_rep])

                    tp_concat_encoding = torch.mean(context_encoding, dim=1) + torch.mean(column_encoding, dim=1)
                    # Norm 추가
                    tp_concat_encoding = table_encoder.norm(tp_concat_encoding)
                    
                    with open(os.path.join(table_index_dir, '{}.faiss'.format(tid)), 'wb') as f:
                        #torch.save(tp_concat_encoding.detach().numpy(), f)
                        torch.save(tp_concat_encoding.cpu().numpy(), f)


def add_generic_arguments(parser):
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir.")
    parser.add_argument("--ckpt_file", default=None, type=str, required=True,
                        help="The ckpt file ")
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--qrel_file", default=None, type=str, required=True,
                        help="The qrel file  file ")
    parser.add_argument("--result_file", default=None, type=str, required=True,
                        help="The result file ")
    parser.add_argument("--index", default='l2', type=str, required=True,
                        help="The index method")



def get_query_dict(args):
    query_dict = dict()
    with open(os.path.join(args.data_dir, 'test.jsonl'))as f:
        for line in f.readlines():
            if line.strip() == '':
                break
            json_data = json.loads(line)
            query_dict[json_data['qid']] = json_data['query']
    return query_dict


def get_qrel_dict(args):
    qrel_dict = dict()
    with open(os.path.join(args.data_dir, 'test.jsonl.qrels'))as f:
        for line in f.readlines():
            if line.strip() == '':
                break
            qid, _, tid, rel = line.split('\t')
            tid = tid.replace('/', '@') # TID보정
            try:
                qrel_dict[qid].append([tid, rel.strip(), 0])
            except:
                qrel_dict[qid] = [[tid, rel.strip(), 0]]
    return qrel_dict


def save_TREC_format(args, ranked_dict):
    with open(args.result_file, 'w', encoding='utf-8') as f:
        # Qid순을 오름차순
        for qid, results in sorted(ranked_dict.items(), key=(lambda x: int(x[0]))):
            # Score기준 내림차순
            sorted_results = sorted(results, key = lambda x : x[-1], reverse=True)
            for idx, result in enumerate(sorted_results):
                tid, rel, score = result
                tid = tid.replace('@', '/') # TID보정
                if args.index == 'l2':
                    f.write(f"{qid}\t{0}\t{tid}\t{rel}\t{idx + 1}\tfaiss\n")
                elif args.index == 'dot':
                    f.write(f"{qid}\t{0}\t{tid}\t{rel}\t{score}\tfaiss\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_arguments(parser)
    QueryTableMatcher.add_model_specific_args(parser)
    args = parser.parse_args()

    # Query 와 Qrel dict를 가져옴
    query_dict = get_query_dict(args)
    qrel_dict = get_qrel_dict(args)

    # Faiss로 search
    ranked_dict = faiss_search(args, query_dict, qrel_dict)

    # TREC format으로 저장
    save_TREC_format(args, ranked_dict)

    print(">>> Run TREC Script ... ", end='')
    trec_eval = TREC_evaluator(qrels_file=args.qrel_file,
                               result_file=args.result_file,
                               trec_cmd="../trec_eval")

    print("Done...")
    print(trec_eval.get_ndcgs(metric='map'))
    print(trec_eval.get_ndcgs(metric='recip_rank'))
    print(trec_eval.get_ndcgs())


