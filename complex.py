import openai
import json
import numpy as np
import textwrap
import re
from time import time,sleep

def open_file(filepath):
    """ It returns the text in the file referenced by the path received. """
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Now, we get the key for the API.
openai.api_key = open_file('key.txt')

# It handles openai's embedding service.
def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII',errors='ignore').decode()
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector

# Dot product is used to quantify the _similarity_ between two vectors.
def similarity(v1, v2):
    return np.dot(v1, v2)

def search_index(text, data, count=10):
    """The input is a query (text), and a set of fragments of text (data). The goal is to choose the _count_ most useful fragments to answer to the query."""
    # The query's embedding is found.
    vector = gpt3_embedding(text)
    scores = list()
    # The similarity to the query is found for fragment in data.
    # Both the similarity and the fragment are saved in a dictionar.
    for i in data:
        score = similarity(vector, i['vector'])
        #print(score)
        scores.append({'content': i['content'], 'score': score})
    # Only the _count_ most fragments with highest similarity are returned.
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    return ordered[0:count]

# It manages the openai's completeions service.
def gpt3_completion(prompt, engine='text-davinci-002', temp=0.6, top_p=1.0, tokens=2000, freq_pen=0.25, pres_pen=0.0, stop=['<<END>>']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('\s+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


if __name__ == '__main__':
    # The received json contains a text fragments of a size accepted by ChatGPT.
    with open('index.json', 'r') as infile:
        data = json.load(infile)
    #print(data)
    while True:
        query = input("Escribe aquí tu pregunta: ")
        #print(query)
        results = search_index(query, data)
        #print(results)
        answers = list()
        # Answer the same question for all returned chunks.
        for result in results:
            prompt = open_file('prompt_answer.txt').replace('<<PASSAGE>>', result['content']).replace('<<QUERY>>', query)
            answer = gpt3_completion(prompt)
            print('\n\n', answer)
            answers.append(answer)
        # Summarize the answers together
        all_answers = '\n\n'.join(answers)
        chunks = textwrap.wrap(all_answers, 10000)
        final = list()
        for chunk in chunks:
            prompt = open_file('prompt_summary.txt').replace('<<SUMMARY>>', chunk)
            summary = gpt3_completion(prompt)
            final.append(summary)
        print('\n\n=========\n\n', '\n\n'.join(final))
