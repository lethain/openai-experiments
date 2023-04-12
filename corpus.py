import os
import numpy as np
import openai
import pandas as pd
import pickle
import tiktoken
import re


COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"




def ask_prompt(prompt):
    templated_prompt = f"""Answer the question as truthfully as possible, and if you're unsure of the answer, say "Sorry, I don't know".

    Q: {prompt}
    A:
    """
    
    resp = openai.Completion.create(
        prompt=templated_prompt,
        temperature=0,
        max_tokens=300,
        model=COMPLETIONS_MODEL
    )["choices"][0]["text"].strip(" \n")
    return resp


def add_embeddings():
    writing_directories = (
        '/Users/will/irrational_hugo/content',
        '/Users/will/infra-eng/content',
        '/Users/will/staff-eng//src/markdown/',
    )
    filepaths = get_filepaths(writing_directories)

    for filepath in filepaths[15:17]:
        clean_entry(filepath)


def clean_entry(filepath):
    raw = open(filepath).read()
    headings = {'filepath': filepath}
    seen_break = 0
    raw_header, body = raw.split('---', 2)[1:]
    for raw_line in raw_header.split('\n'):
        line = raw_line.strip()
        if ':' in line:
            key, val = line.split(':', 1)
            headings[key] = val.strip(" \"'")

    title = headings['title'] if 'title' in headings else filepath
    body = f"# {headings['title']}" + body

    print('headings', headings)
    
    sections = re.findall("[#]{1,4} .*\n", body)
    print('sections')
    for section in sections:
        print(section)




def get_filepaths(directories):
    fps = []
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if not file_path.endswith('~'):
                    fps.append(file_path)
    return fps


if __name__ == "__main__":
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        key_file = os.path.join(os.path.expanduser("~"), ".openai-api-key.txt")
        api_key = open(key_file).read().strip()

    openai.api_key = api_key

    add_embeddings()

    #resp = ask_prompt("What are 10 things I should do on a sunny day in San Francisco?")
    #print(resp)
