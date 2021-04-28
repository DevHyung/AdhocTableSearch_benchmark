import itertools
import json
import os
import random
from math import ceil
from collections import defaultdict
from pathlib import Path
import re

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

from table_bert import Table, Column, TableBertModel


class Sample(object):
    def __init__(self, query, positive_tables, negative_tables):
        self.query = query
        self.positive_tables = positive_tables
        self.negative_tables = negative_tables


class QueryTableDataset(Dataset):
    def __init__(self, data_dir: str = '.data', data_type: str = 'train',
                 query_tokenizer=None, table_tokenizer=None, max_query_length=7,
                 prepare=False, is_slice=False):
        self.data_dir = data_dir
        self.query_file = data_type + '.query'
        self.table_file = data_type + '.table'
        self.ids_file = data_type + '.pair'
        self.data_type = data_type  # test, train 구분하기위해
        self.is_slice = is_slice
        if prepare:
            self.prepare(data_dir, data_type, query_tokenizer, table_tokenizer, max_query_length)

        self.data = torch.load(os.path.join(self.processed_folder, self.ids_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def slice_table(self, title, heading, datas, caption, table_tokenizer, query, rel):
        table_rep_list = []

        min_row = 10  # 최소 5개의 행은 있어야 함
        max_table_nums = 10  # 테이블은 최대 10개로 나뉘어짐

        # TODO: max_table_nums = 2, 5, 10 으로 바꿔보면서 테스트
        if len(datas) <= min_row:  # 테이블이 최소행 보다 작은 경우
            column_rep = Table(id=title,
                               header=[Column(h.strip(), 'text') for h in heading],
                               data=datas
                               ).tokenize(table_tokenizer)
            table_rep_list.append((rel, column_rep))
        else:
            row_n = max(min_row, ceil(len(datas) / max_table_nums))
            slice_row_data = [datas[i * row_n:(i + 1) * row_n] for i in range((len(datas) + row_n - 1) // row_n)]
            if str(rel) == 0:  # Negative
                for rows in slice_row_data:
                    column_rep = Table(id=title,
                                       header=[Column(h.strip(), 'text') for h in heading],
                                       data=rows
                                       ).tokenize(table_tokenizer)
                    table_rep_list.append((rel, column_rep))

            else:  # Positive
                query_tokens = [token.strip() for token in query.split(' ')]
                is_always_postive = False
                for token in query_tokens:
                    if token in caption:
                        is_always_postive = True
                        break
                if is_always_postive:  # caption에 포함되어있는 경우
                    for rows in slice_row_data:
                        column_rep = Table(id=title,
                                           header=[Column(h.strip(), 'text') for h in heading],
                                           data=rows
                                           ).tokenize(table_tokenizer)
                        table_rep_list.append((rel, column_rep))
                else:
                    for rows in slice_row_data:
                        column_rep = Table(id=title,
                                           header=[Column(h.strip(), 'text') for h in heading],
                                           data=rows
                                           ).tokenize(table_tokenizer)
                        modify_rel = '0'
                        # Row data를 하나의 string으로
                        cell_string_sum = ''
                        for row in rows:
                            cell_string_sum += ' '.join(row)
                        # Query tokens과 overlap
                        for token in query_tokens:
                            if token in cell_string_sum:
                                modify_rel = '1'
                                break
                        table_rep_list.append((modify_rel, column_rep))

        return table_rep_list

    def infer_column_type_from_row_values(self, numeric_idx_list, heading, body):
        heading_type_dict = {k: 'text' for k in heading}
        for n_idx in numeric_idx_list:
            heading_type_dict[heading[n_idx]] = 'real'
            for i, rows in enumerate(body):
                try:
                    float(rows[n_idx].strip().replace('−', '-').replace(',', '').replace('–', '-'))
                except:
                    heading_type_dict[heading[n_idx]] = 'text'
                    break
        return heading_type_dict

    def prepare(self, data_dir, data_type, query_tokenizer, table_tokenizer, max_query_length):
        if self._check_exists():
            return

        processed_dir = Path(self.processed_folder)
        processed_dir.mkdir(exist_ok=True)
        if not (query_tokenizer and table_tokenizer):
            raise RuntimeError('Tokenizers are not found.' +
                               ' You must set query_tokenizer and table_tokenizer')
        print('Processing...')

        query_dict = defaultdict()
        pos_tables, neg_tables = defaultdict(list), defaultdict(list)
        data = []
        path = Path(data_dir + '/' + data_type + '.jsonl')

        with open(path) as f:
            for line in f.readlines():
                if not line.strip():
                    break
                # 테이블 Meta data parsing ( qid, tid, query, rel )
                jsonStr = json.loads(line)
                query = jsonStr['query']
                qid = jsonStr['qid']
                tid = table_json['docid']

                # Query Encode
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

                # Table Encode
                caption_rep, column_reps = encode_tables(jsonStr, self.is_slice, query, table_tokenizer)

                for (rel, column_rep) in column_reps:
                    if str(rel) == '0':
                        neg_tables[qid].append((column_rep, caption_rep))
                    else:
                        pos_tables[qid].append((column_rep, caption_rep))

        for qid in query_dict:
            if not pos_tables[qid]:
                continue
            for t in itertools.product(pos_tables[qid], neg_tables[qid]):
                data.append([query_dict[qid]] + list(itertools.chain.from_iterable(t)))

        # Save
        with open(os.path.join(processed_dir, self.ids_file), 'wb') as f:
            torch.save(data, f)
        print('Done!')

    @property
    def processed_folder(self):
        return os.path.join(self.data_dir, 'processed')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.ids_file))

def encode_tables(table_json, is_slice, query, table_tokenizer):
    rel = table_json['rel']

    html_pattern = re.compile(r'<\w+ [^>]*>([^<]+)</\w+>')
    tag_pattern = re.compile(r'<.*?>')
    link_pattern = re.compile(r'\[.*?\|.*?\]')

    # Raw Json parsing ( Detail table information )
    raw_json = json.loads(table_json['table']['raw_json'])

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
    caption_rep = table_tokenizer.tokenize(caption)

    if is_slice:
        column_reps = slice_table(title, heading, body, caption, table_tokenizer, query, rel)

    else:
        column_reps = [(rel,
                        Table(id=caption,
                              header=[Column(h.strip(), 'text') for h in heading],
                              data=body
                              ).tokenize(table_tokenizer))]
    return caption_rep, column_reps

def slice_table( title, heading, datas, caption, table_tokenizer, query, rel):
    table_rep_list = []

    min_row = 10  # 최소 5개의 행은 있어야 함
    max_table_nums = 10  # 테이블은 최대 10개로 나뉘어짐

    # TODO: max_table_nums = 2, 5, 10 으로 바꿔보면서 테스트
    if len(datas) <= min_row:  # 테이블이 최소행 보다 작은 경우
        column_rep = Table(id=title,
                           header=[Column(h.strip(), 'text') for h in heading],
                           data=datas
                           ).tokenize(table_tokenizer)
        table_rep_list.append((rel, column_rep))
    else:
        row_n = max(min_row, ceil(len(datas) / max_table_nums))
        slice_row_data = [datas[i * row_n:(i + 1) * row_n] for i in range((len(datas) + row_n - 1) // row_n)]
        if str(rel) == 0:  # Negative
            for rows in slice_row_data:
                column_rep = Table(id=title,
                                   header=[Column(h.strip(), 'text') for h in heading],
                                   data=rows
                                   ).tokenize(table_tokenizer)
                table_rep_list.append((rel, column_rep))

        else:  # Positive
            query_tokens = [token.strip() for token in query.split(' ')]
            is_always_postive = False
            for token in query_tokens:
                if token in caption:
                    is_always_postive = True
                    break
            if is_always_postive:  # caption에 포함되어있는 경우
                for rows in slice_row_data:
                    column_rep = Table(id=title,
                                       header=[Column(h.strip(), 'text') for h in heading],
                                       data=rows
                                       ).tokenize(table_tokenizer)
                    table_rep_list.append((rel, column_rep))
            else:
                for rows in slice_row_data:
                    column_rep = Table(id=title,
                                       header=[Column(h.strip(), 'text') for h in heading],
                                       data=rows
                                       ).tokenize(table_tokenizer)
                    modify_rel = '0'
                    # Row data를 하나의 string으로
                    cell_string_sum = ''
                    for row in rows:
                        cell_string_sum += ' '.join(row)
                    # Query tokens과 overlap
                    for token in query_tokens:
                        if token in cell_string_sum:
                            modify_rel = '1'
                            break
                    table_rep_list.append((modify_rel, column_rep))

    return table_rep_list

def query_table_collate_fn(batch):
    query, pos_column, pos_caption, neg_column, neg_caption = zip(*batch)
    input_ids, token_type_ids, attention_mask = [], [], []
    for q in query:
        input_ids.append(q["input_ids"].squeeze())
        token_type_ids.append(q["token_type_ids"].squeeze())
        attention_mask.append(q["attention_mask"].squeeze())

    query = {"input_ids": torch.stack(input_ids),
             "token_type_ids": torch.stack(token_type_ids),
             "attention_mask": torch.stack(attention_mask)}
    return query, pos_column, pos_caption, neg_column, neg_caption


class QueryTablePredictionDataset(Dataset):
    def __init__(self, data_dir: str = '.data', data_type: str = 'test',
                 query_tokenizer=None, table_tokenizer=None, max_query_length=27,
                 prepare=False, is_slice=False):
        self.data_dir = data_dir
        self.query_file = data_type + '.query'
        self.table_file = data_type + '.table'
        self.ids_file = data_type + '.pair'
        self.is_slice = is_slice

        if prepare:
            self.prepare(data_dir, data_type, query_tokenizer, table_tokenizer, max_query_length)

        self.pair_ids = torch.load(os.path.join(self.processed_folder, self.ids_file))

    def __len__(self):
        return len(self.pair_ids)

    def __getitem__(self, index):
        return self.pair_ids[index]

    def slice_table(self, title, heading, datas, table_tokenizer):
        #  for prediction
        #  따로 슬라이싱을해서 rel을 수정 할 필요가 없음
        table_rep_list = []

        min_row = 5  # 최소 5개의 행은 있어야 함
        max_table_nums = 10  # 테이블은 최대 10개로 나뉘어짐

        if len(datas) <= min_row:
            column_rep = Table(id=title,
                               header=[Column(h.strip(), 'text') for h in heading],
                               data=datas
                               ).tokenize(table_tokenizer)
            table_rep_list.append(column_rep)
        else:
            row_n = max(min_row, ceil(len(datas) / max_table_nums))
            slice_row_data = [datas[i * row_n:(i + 1) * row_n] for i in range((len(datas) + row_n - 1) // row_n)]
            for rows in slice_row_data:
                column_rep = Table(id=title,
                                   header=[Column(h.strip(), 'text') for h in heading],
                                   data=rows
                                   ).tokenize(table_tokenizer)
                table_rep_list.append(column_rep)
        return table_rep_list

    def infer_column_type_from_row_values(self, numeric_idx_list, heading, body):
        heading_type_dict = {k: 'text' for k in heading}
        for n_idx in numeric_idx_list:
            heading_type_dict[heading[n_idx]] = 'real'
            for i, rows in enumerate(body):
                try:
                    float(rows[n_idx].strip().replace('−', '-').replace(',', '').replace('–', '-'))
                except:
                    heading_type_dict[heading[n_idx]] = 'text'
                    break
        return heading_type_dict

    def prepare(self, data_dir, data_type, query_tokenizer, table_tokenizer, max_query_length):
        if self._check_exists():
            return

        processed_dir = Path(self.processed_folder)
        processed_dir.mkdir(exist_ok=True)
        if not (query_tokenizer and table_tokenizer):
            raise RuntimeError('Tokenizers are not found.' +
                               ' You must set query_tokenizer and table_tokenizer')
        print('Processing...')

        query_dict = defaultdict()
        pairs = []
        path = Path(data_dir + '/' + data_type + '.jsonl')

        with open(path) as f:
            for line in f.readlines():
                if not line.strip():
                    break

                # 테이블 Meta data parsing ( qid, tid, query, rel )
                jsonStr = json.loads(line)
                tid = jsonStr['docid']
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

                # Table Encode
                caption_rep, column_reps = encode_tables(jsonStr, self.is_slice, query, table_tokenizer)

                for column_rep in column_reps:
                    pairs.append([qid, query_dict[qid], tid, column_rep, caption_rep, rel])

        # Save
        with open(os.path.join(processed_dir, self.ids_file), 'wb') as f:
            torch.save(pairs, f)
        print('Done!')

    @property
    def processed_folder(self):
        return os.path.join(self.data_dir, 'processed')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.ids_file))


def query_table_prediction_collate_fn(batch):
    qid, query, tid, column, caption, rel = zip(*batch)
    input_ids, token_type_ids, attention_mask = [], [], []
    for q in query:
        input_ids.append(q["input_ids"].squeeze())
        token_type_ids.append(q["token_type_ids"].squeeze())
        attention_mask.append(q["attention_mask"].squeeze())

    query = {"input_ids": torch.stack(input_ids),
             "token_type_ids": torch.stack(token_type_ids),
             "attention_mask": torch.stack(attention_mask)}
    return query, column, caption, rel, qid, tid


if __name__ == "__main__":
    query_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    table_model = TableBertModel.from_pretrained('model/tabert_base_k3/model.bin')
    table_tokenizer = table_model.tokenizer

    dataset = QueryTableDataset(data_dir='data/1',
                                data_type='train',
                                query_tokenizer=query_tokenizer,
                                table_tokenizer=table_tokenizer,
                                prepare=True,
                                )
    dataloader = DataLoader(dataset,
                            batch_size=2,
                            collate_fn=query_table_collate_fn)

    for _ in range(1):
        for d in dataloader:
            print(d)
            break