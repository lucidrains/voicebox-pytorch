import argparse
from multiprocessing import Process
import os
import pandas as pd
import numpy as np
from pandarallel import pandarallel
from collections import defaultdict
import textgrid
from tqdm import tqdm
import shutil

"""
prepare command
python data/generate_alignments.py \
    --command prepare \
    --dataset /home/jovyan/data/speechtospeech/cvss_t_processed_de_es_fr_ca \
    --num-procs 80
  
sync command
python data/generate_alignments.py \
    --command sync \
    --dataset /home/jovyan/data/speechtospeech/cvss_t_processed_de_es_fr_ca \
    --num-procs 80
    

mfa command
/home/jovyan/.conda/envs/aligner/bin/mfa align --clean --single_speaker --num_jobs 10 /home/jovyan/data/speechtospeech/MFA_DATA/input_data \
    /home/jovyan/data/speechtospeech/MFA_DATA/dictionary.txt \
    english_us_arpa \
    /home/jovyan/data/speechtospeech/MFA_DATA/output_textgrids \
    --silence_probability .7

"""

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--command', type=str, help='this should be "prepare" or "sync"', required=True)
    # prepare if you haven't yet run the MFA
    # if you ran the MFA, do sync
    parser.add_argument('--dataset', type=str,
                        help='directory the dataset is in that we want to run this on.')
    parser.add_argument('--max-entries', type=int, default=-1,
                       help='for debugging: max entries to run on per split')
    parser.add_argument('--num-procs', type=int, default=16)

    args = parser.parse_args()
    return args

def get_wav_fname(row, dataset, split):
    if 'source' in row:
        return os.path.join(row['source'], 'output', split, row['id'] + '.wav')
    else:
        return os.path.join(dataset, 'output', split, row['id'] + '.wav')

def prep_sample(row, dest):
    name = row['wav'].split('/')[-1][:-4]
    src_name = row['source'].split('/')[-1]
    shutil.copy(row['wav'], os.path.join(dest, src_name + '.' + name + '.wav'))
    with open(os.path.join(dest, src_name + '.' + name + '.txt'), 'w') as f:
        f.write(row['text'])
        f.close()
        

def make_dictionary(df): # expect columns text and phonemes
    dictionary = defaultdict(lambda: defaultdict(lambda: 0))
    for i, row in tqdm(df.iterrows()):
        tokens = row['text'].split(' ')
        phone_chunks = row['phonemes'][8:-6].split(' _ ')

        phone_chunks = list(filter(lambda x: x!="'", phone_chunks))
        if len(phone_chunks)!=len(tokens):
            print(i)
            print(phone_chunks)
            print(tokens)
            continue
        for tok, phones in zip(tokens, phone_chunks):
            dictionary[tok][phones] += 1
    dictionary['<start>'] = {'sil': 1}
    dictionary['<eos>'] = {'sil': 1}
    return dictionary

def save_dictionary(fname, dictionary):
    output = ""
    for tok, d in dictionary.items():
        total_occurrences = sum(list(d.values()))
        for phones, occ in d.items():
            output += tok + '\t' + str(occ/total_occurrences) + '\t' + phones + '\n'
    with open(fname, 'w') as f:
        f.write(output)
        f.close()

def sync_up(grid, phoneme_str):
    phonemes = phoneme_str.split(' ')
    times = []
    synced = []
    grid_ctr = 0
    was_issue = False
    for i, phoneme in enumerate(phonemes):
        if grid_ctr>=len(grid):
            guess = None
            guess_dur = .001
            if i+1<len(phonemes):
                was_issue = True
                print("AAAHHHHH FAILLUURREEE")
        else:
            guess = grid[grid_ctr].mark
            guess_dur = grid[grid_ctr].duration()
        if phoneme in ['<start>', '_', '<eos>']: # if it's a whitespace phoneme...
            if guess=='':
                times.append(guess_dur)
                synced.append(guess)
                grid_ctr+=1
            else:
                times.append(.001) # negligible silence
                synced.append(guess)
        else: # assume alignment otherwise
            times.append(guess_dur)
            synced.append(guess)
            grid_ctr+=1
    return times, synced, was_issue

if __name__=='__main__':
    args = parse_args()
    mfa_data_path = '/home/jovyan/data/speechtospeech/MFA_DATA'
    
    if args.command=='prepare':

        # make directories
        if os.path.exists(mfa_data_path):
            shutil.rmtree(mfa_data_path)
        os.mkdir(mfa_data_path)
        os.mkdir(os.path.join(mfa_data_path, 'input_data'))
        os.mkdir(os.path.join(mfa_data_path, 'output_textgrids'))

        # acquire data to start
        all_mfa_data = pd.DataFrame()

        split_names = [x[:-4] for x in os.listdir(args.dataset) if x.endswith('.tsv') and 'alignments' not in x]
        all_dfs = []
        for split in split_names:
            df = pd.read_csv(os.path.join(args.dataset, split + '.tsv'), index_col=None, sep='\t')

            # for debugging, sample subset of df
            if args.max_entries>0:
                df = df.sample(n=args.max_entries, replace=False)

            # first get wav fnames
            pandarallel.initialize(progress_bar=True)
            wav_fnames = df.parallel_apply(lambda x: get_wav_fname(x, args.dataset, split), axis=1)

            if 'source' in df.columns:
                sources = df['source']
            else:
                sources = ['' for _ in range(len(df))]

            newdf = pd.DataFrame({'text': df['text'], 'phonemes': df['phonemes'], 'wav': wav_fnames, 'source': sources, 'split': split, 'id': df['id']})
            all_mfa_data = pd.concat([newdf, all_mfa_data])

        # copy stuff over
        pandarallel.initialize(progress_bar=True)
        all_mfa_data.parallel_apply(lambda x: prep_sample(x, os.path.join(mfa_data_path, 'input_data')), axis=1)

        # build and save dictionary
        dictionary = make_dictionary(all_mfa_data)
        save_dictionary(os.path.join(mfa_data_path, 'dictionary.txt'), dictionary)
        
        all_mfa_data.to_csv(os.path.join(mfa_data_path, 'all_mfa_data.csv'), index=False)

    elif args.command=='sync':
        split_names = [x[:-4] for x in os.listdir(args.dataset) if x.endswith('.tsv') and 'alignments' not in x]
        
        all_mfa_data = pd.read_csv(os.path.join(mfa_data_path, 'all_mfa_data.csv'), index_col=None)
        # read the textgrids
        print('Reading textgrids')
        textgrids = {fname[:-9]:textgrid.TextGrid.fromFile(os.path.join(mfa_data_path, 'output_textgrids', fname)) for fname in
                 tqdm(os.listdir(os.path.join(mfa_data_path, 'output_textgrids')))}
        
        all_mfa_data = all_mfa_data[all_mfa_data.apply(lambda x: x['source'].split('/')[-1] + '.' + x['id'] in textgrids, axis=1)]

        # now we gotta sync everything
        print('Syncing alignment results')
        pandarallel.initialize(progress_bar=True)
        alignments = all_mfa_data.parallel_apply(lambda x: sync_up(textgrids[x['source'].split('/')[-1] + '.' + x['id']][1], x['phonemes']), axis=1)

        # store in a df
        print('Saving everything')
        all_mfa_data['alignment_issue'] = alignments.apply(lambda x: x[2])
        all_mfa_data['durations'] = alignments.apply(lambda x: x[0])
        all_mfa_data['synced_phonemes'] = alignments.apply(lambda x: x[1])

        for split in split_names:
            all_mfa_data[all_mfa_data['split']==split].to_csv(os.path.join(args.dataset, f'{split}_alignments.tsv'), index=False, sep='\t')
    
    else:
        raise NotImplementedError(args.command + ' is not supported')
    