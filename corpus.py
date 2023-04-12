import os
import numpy as np
import openai
import pandas as pd
import pickle
import tiktoken
import re


COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"
MAX_EMBEDDINGS = 1536
EMBEDDINGS_CACHE = None
EMBEDDINGS_CACHE_FILE = "embeddings.pkl"
EMBEDDINGS_CSV = "embeddings.csv"


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

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.

    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities


def get_document_embeddings():
    df = get_all_embeddings()
    max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading"])
    return {
        (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }


def get_all_embeddings():
    try:
        df = pd.read_csv(EMBEDDINGS_CSV, header=0)
        return df
    except Exception as e:
        print("Couldn't read DF from disk", e)

    writing_directories = (
        '/Users/will/irrational_hugo/content',
        '/Users/will/infra-eng/content',
        '/Users/will/staff-eng//src/markdown/',
    )
    filepaths = get_filepaths(writing_directories)

    rows = []
    for filepath in filepaths:
        try:
            headers, sections = clean_entry(filepath)
            for sect_title, sect_content in sections:
                row = {
                    'title': headers['filename'],
                    'heading': sect_title,
                    'content': sect_content,
                }
                rows.append(row)
        except Exception as e:
            print(filepath, e)

    df = pd.DataFrame(rows, columns=['title', 'heading', 'content'])
    print(df.sample(5))

    embeddings = compute_doc_embeddings(df)

    cols = ('title', 'heading') + tuple(range(MAX_EMBEDDINGS))
    export_rows = []
    for emb in embeddings:
        new_row = [emb['title'], emb['heading']]
        for i in range(MAX_EMBEDDINGS):
            new_row.append(emb['idx'][i])
        export_rows.append(new_row)
    export_df = pd.DataFrame(export_rows, columns=cols)
    export_df.to_csv(EMBEDDINGS_CSV, index=False)

    return export_df


def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    global EMBEDDINGS_CACHE
    if EMBEDDINGS_CACHE is None:
        try:
            EMBEDDINGS_CACHE = pd.read_pickle(EMBEDDINGS_CACHE_FILE)
        except (FileNotFoundError, EOFError):
            EMBEDDINGS_CACHE = {}

    key = (model, text)
    if key not in EMBEDDINGS_CACHE:
        result = openai.Embedding.create(
            model=model,
            input=text
        )
        EMBEDDINGS_CACHE[key] = result["data"][0]["embedding"]

        with open(EMBEDDINGS_CACHE_FILE, "wb") as embedding_cache_file:
            pickle.dump(EMBEDDINGS_CACHE, embedding_cache_file)

    return EMBEDDINGS_CACHE[key]
    

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    rows = []
    for idx, r in df.iterrows():
        try:
            row = {
                'title': r.title,
                'heading': r.heading,
                'idx': get_embedding(r.content)
            }
            rows.append(row)
        except openai.error.InvalidRequestError as ire:
            print(r.title, r.heading, ire)

    return rows


def clean_entry(filepath):
    """
    Parse a Markdown file with metadata headers wrapped in ---, e.g. a Hugo blog post.
    Return a 2-tuple of headers dictionary, with additional filepath key,
    and a list of header-contents 2-tuples.
    """
    raw = open(filepath).read()
    headings = {
        'filepath': filepath,
        'filename': filepath.split('/')[-1],
    }
    seen_break = 0
    if raw.startswith('---'):
        raw_header, body = raw.split('---', 2)[1:]
        for raw_line in raw_header.split('\n'):
            line = raw_line.strip()
            if ':' in line:
                key, val = line.split(':', 1)
                headings[key] = val.strip(" \"'")
    else:
        body = raw

    title = headings['title'] if 'title' in headings else headings['filename']
    body = f"# {title}\n{body}"

    sections = re.findall("[#]{1,4} .*\n", body)
    split_txt = '##### !!'
    for section in sections:
        body = body.replace(section, split_txt)
    contents = [x.strip() for x in body.split(split_txt)]
    headers = [x.strip('# \n') for x in sections]
    combined = zip(headers, contents)
    # strip out very short content sections
    combined = [(x,y) for x,y in combined if len(y.strip()) > 40]
    return headings, combined


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

    document_embeddings = get_document_embeddings()
    x = order_document_sections_by_query_similarity("How should I get an engineering executive job?", document_embeddings)[:5]
    print(x)


    #max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading"])
    #return {
    #(r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()


    #resp = ask_prompt("What are 10 things I should do on a sunny day in San Francisco?")
    #print(resp)
