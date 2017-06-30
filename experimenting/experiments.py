import re

def find_companies():
    import spacy
    nlp = spacy.load('en_core_web_md')

    with open('/tmp/spacy_nyse.txt', 'w', encoding='utf8') as out_file:
        with open('/tmp/sentence_nyse.tsv', 'r', encoding='utf8') as in_file:
            for line in in_file:
                if len(line.split('\t')) > 2:
                    continue
                [sentence, companies] = line.split('\t')
                doc = nlp(sentence)
                out_file.write('\n' + sentence + '\n')

                for ent in doc.ents:
                    if ent.label_ != "ORG":
                        continue
                    out_file.write('\t'.join([ent.text]) + '\n')

                if len(companies) == 0:
                    continue
                spacy_companies = [ent.text for ent in doc.ents]
                for company in companies.split(','):
                    if company.strip() not in map(lambda name: name.lower(), spacy_companies):
                        out_file.write('\t'.join([company.strip()]) + '\n')


def separate_sentence_data(lines):
    sub_list = []
    lines_splited = []
    for e in lines:
        if e == '':
            if sub_list:
                lines_splited.append(sub_list)
            sub_list = [e]
        else:
            sub_list.append(e)
    lines_splited.append(sub_list)
    return [list(filter(None, list_item)) for list_item in lines_splited]


def get_occurance_tuple(company, sentence):
    company_occurances = []
    for m in re.finditer(r'\b' + re.escape(company) + r'\b', sentence):
        company_occurances.append((m.start(), m.end(), 'ORG'))
    return company_occurances


def get_training_data(lines_split):
    training_data = []
    for sentence_data in lines_split:
        company_tuples = []
        for company in sentence_data[1:]:
            [company_tuples.append(company_tuple) for company_tuple in get_occurance_tuple(company, sentence_data[0])]
        training_data.append((sentence_data[0], company_tuples))
    return training_data


def make_ner_training_file():
    training_data = []
    with open('../data/spacy_nyse.txt', 'r', encoding='utf8') as in_file:
        lines_split = [line.strip() for line in in_file]
    lines_split = separate_sentence_data(lines_split)
    return get_training_data(lines_split)


# make_ner_training_file()

import spacy
import random
from spacy.gold import GoldParse
from spacy.language import EntityRecognizer


def ner_test(model):
    nlp = spacy.load(model)

    doc = nlp(
        '5.11 is a portfolio company of Compass Diversified Holdings (NYSE: CODI).')

    for ent in doc.ents:
        if ent.label_ != "ORG":
            continue
        print('\t'.join([ent.text]))


def ner_test_after():
    spacy.util.set_data_path('/tmp/model/')
    nlp = spacy.load('en_core_web_md_t')

    doc = nlp(
        'ET AL lowered its stake in American Express Company (NYSE:AXP) by 2.9% during the fourth quarter, Holdings Channel reports.')

    for ent in doc.ents:
        if ent.label_ != "ORG":
            continue
        print('\t'.join([ent.text]))


def ner_train(model):
    train_data = make_ner_training_file()

    nlp = spacy.load(model)
    ner = EntityRecognizer(nlp.vocab, entity_types=['ORG'])

    for itn in range(5):
        random.shuffle(train_data)
        for raw_text, entity_offsets in train_data:
            doc = nlp.make_doc(raw_text)
            gold = GoldParse(doc, entities=entity_offsets)

            nlp.tagger(doc)
            ner.update(doc, gold)
    ner.model.end_training()
    nlp.save_to_directory('/tmp/model/' + model + '_t')


# ner_test('en_core_web_md')
# ner_train('en_core_web_md')
# ner_test_after()

# nlp = spacy.load('en_core_web_md', entity=False, parser=False)

# doc = nlp('ET AL lowered its stake in American Express Company (NYSE:AXP) by 2.9% during the fourth quarter, Holdings Channel reports.')
#
# for ent in doc.ents:
#     if ent.label_ != "ORG":
#         continue
#     print('\t'.join([ent.text]) + '\n')






# from __future__ import unicode_literals, print_function
import json
import pathlib
import random

import spacy
from spacy.pipeline import EntityRecognizer
from spacy.gold import GoldParse
from spacy.tagger import Tagger

try:
    unicode
except:
    unicode = str


def train_ner(nlp, train_data, entity_types):
    # Add new words to vocab.
    for raw_text, _ in train_data:
        doc = nlp.make_doc(raw_text)
        for word in doc:
            _ = nlp.vocab[word.orth]

    # Train NER.
    ner = EntityRecognizer(nlp.vocab, entity_types=entity_types)
    for itn in range(5):
        random.shuffle(train_data)
        for raw_text, entity_offsets in train_data:
            doc = nlp.make_doc(raw_text)
            gold = GoldParse(doc, entities=entity_offsets)
            ner.update(doc, gold)
    return ner


def save_model(ner, model_dir):
    model_dir = pathlib.Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir()
    assert model_dir.is_dir()

    with (model_dir / 'config.json').open('wb') as file_:
        data = json.dumps(ner.cfg)
        if isinstance(data, unicode):
            data = data.encode('utf8')
        file_.write(data)
    ner.model.dump(str(model_dir / 'model'))
    if not (model_dir / 'vocab').exists():
        (model_dir / 'vocab').mkdir()
    ner.vocab.dump(str(model_dir / 'vocab' / 'lexemes.bin'))
    with (model_dir / 'vocab' / 'strings.json').open('w', encoding='utf8') as file_:
        ner.vocab.strings.dump(file_)


def main(model_dir=None):
    nlp = spacy.load('en_core_web_md', parser=False, entity=False, add_vectors=False)

    # v1.1.2 onwards
    if nlp.tagger is None:
        print('---- WARNING ----')
        print('Data directory not found')
        print('please run: `python -m spacy.en.download --force all` for better performance')
        print('Using feature templates for tagging')
        print('-----------------')
        nlp.tagger = Tagger(nlp.vocab, features=Tagger.feature_templates)

    train_data = make_ner_training_file()
    ner = train_ner(nlp, train_data, ['ORG'])

    for sentences_data in train_data:
        print(sentences_data[0])
        doc = nlp.make_doc(sentences_data[0])
        nlp.tagger(doc)
        ner(doc)
        # for word in doc:
        #     print(word.text, word.orth, word.lower, word.tag_, word.ent_type_, word.ent_iob)
        for ent in doc.ents:
            if ent.label_ != "ORG":
                continue
            print('\t'.join([ent.text]))
        print('\n')

    if model_dir is not None:
        save_model(ner, model_dir)

# ner_test('')
if __name__ == '__main__':
    main('ner')
    # Who "" 2
    # is "" 2
    # Shaka "" PERSON 3
    # Khan "" PERSON 1
    # ? "" 2