import spacy
import numpy as np

MAX_K = 80

nlp = None#spacy.load("en_core_web_trf")


def get_chunks(head, chunks):
    children = list(head.subtree)
    if len(children) == 1:
        return
    begin = children[0].idx
    end = children[-1].idx + len(children[-1].text)
    chunks.append((begin, end))
    for child in head.children:
        get_chunks(child, chunks)


def get_all_chunks(text):
    result = []
    global nlp
    if nlp is None:
        nlp=spacy.load("en_core_web_trf")
    doc = nlp(text)
    sentchunks = []
    roots = [token for token in doc if token.head == token]
    for root in roots:
        tmpchunks = []
        get_chunks(root, tmpchunks)
        sentchunks.append(sorted(tmpchunks, key=lambda x: x[1] - x[0], reverse=True))
    
    for i in range(max([len(x) for x in sentchunks])):
        for sentchunk in sentchunks:
            if i < len(sentchunk):
                result.append(sentchunk[i])
                #print(text[sentchunk[i][0]:sentchunk[i][1]])
    
    return result


#print(get_all_chunks("The president is doubling down on an increasingly risky candidate. Justice Anthony Kennedy gave President Donald Trump the biggest political gift possible when he announced his resignation in the spring. Even though not everything that’s gone wrong is Trump’s fault, he has managed to make a complete hash of Brett Kavanaugh’s nomination to fill the empty seat on the court."))

def get_parse_matrix(text, word_offsets):
    result = np.zeros((len(word_offsets), MAX_K))
    chunk_offsets = get_all_chunks(text)
    #print("Finished parsing with "+str(len(chunk_offsets))+" chunks, cutting to "+str(MAX_K))
    if len(chunk_offsets) > MAX_K:
        chunk_offsets = chunk_offsets[0:MAX_K]
    for k in range(len(chunk_offsets)):
        for i in range(len(word_offsets)):
            if word_offsets[i][1] > chunk_offsets[k][0]:
                break
        while i < len(word_offsets):
            if word_offsets[i][0] < chunk_offsets[k][1]:
                result[i][k] = 1
                i=i+1
            else:
                break
    return result
