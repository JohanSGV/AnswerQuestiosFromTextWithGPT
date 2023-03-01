import openai
import json
import textwrap
import time


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


openai.api_key = open_file('key.txt')


def gpt3_embedding(content, engine='text-embedding-ada-002'):
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


if __name__ == '__main__':
    alltext = open_file('el_quijote.txt')
    chunks = textwrap.wrap(alltext, width=4000)
    result = list()
    n_fragments = len(chunks)
    i = 0
    for chunk in chunks:
        time.sleep(4)
        embedding = gpt3_embedding(chunk.encode(encoding='ASCII',errors='ignore').decode())
        info = {'content': chunk, 'vector': embedding}
        result.append(info)
        i += 1
        print(i, "/",n_fragments)
    with open('index.json', 'w') as outfile:
        json.dump(result, outfile, indent=2)
