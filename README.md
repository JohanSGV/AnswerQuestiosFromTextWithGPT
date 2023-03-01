# Answering questions based on a given text using GPT-3

This code employs the code shown on a [David Shapiro's video](https://www.youtube.com/watch?v=es8e4SEuvV0&t=255s).

An outline of tha occurs in `complex.py` is this:

1. The embedding of the text with which questions are going to be answered is found: for this purpose, the text is fragmented into chunks that respect the limit of tokens of GPT-3,
2. The embedding of the question or query is found,
3. Selection  of the most relevant fragments to the question,
4. The question is answered by means of each fragment selected,
5. These multiple questions are summarized and presented as the final answer.

For step 1., `index.py` is used over the text with which to answer questions to get its index: a `json` with the fragmented text.

The prompts used to summarize and ask questions are included (`-ES` means it is the Spanish version).

Finally, for the purpose of testing, the Spanish book *Don Quijote de la Mancha* is provided, as well as its index.
