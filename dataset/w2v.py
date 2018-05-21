# -*- coding: utf-8 -*-




class word2vec: 
    '''
    input:
        wordList(id2word): traverse 
        or 
        wordDict(word2id):
    
    output:
        embedding_dict:
        or
        vocab_vec:
    
    '''
    @staticmethod    
    def load_txt_vec(fname, wordmap):
        """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
        num=0
        word_vecs = {}
        wordvecfile=open(fname,'r')
        for line in wordvecfile:
            num+=1
            if num%1000000==0:
                print ("load words",num)
            line=line.strip()
            elements=re.split('\s+',line)
            word=elements[0]
           # print word
            if word in wordmap:
                vector=[]
                for i in range(1,len(elements)):
                    vector.append(float(elements[i]))
                word_vecs[word]=vector
                
        return word_vecs
    
    @staticmethod
    def load_bin_vec(fname, wordmap):
        """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
        word_vecs = {}
        num=0
        with open(fname, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            print ("word num ", vocab_size)
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
               # word=word.decode('utf8')
                if word in wordmap:
                   #print word
                   word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.read(binary_len)
                num+=1
                if num%1000000==0:
                    print ("load words",num)
                
        return word_vecs

    # from https://github.com/cahya-wirawan/cnn-text-classification-tf/blob/master/data_helpers.py
    @staticmethod
    def load_embedding_vectors_glove(vocabulary, filename, vector_size):
        """
        load embedding_vectors from the glove
        initial matrix with random uniform
        
        pros: 1. fast
        cons: 1. can not get oov
        """
        #embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        embedding_vectors = np.zeros((len(vocabulary), vector_size))
        f = open(filename)
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            idx = vocabulary.get(word)
            if idx != 0:
                embedding_vectors[idx] = vector
        f.close()
        return embedding_vectors
    
    
    @staticmethod
    def load_embedding_vectors_glove_xs(filename, wordmap):
        """
        load embedding_vectors from the glove
        initial matrix with random uniform
        
        pros: 1. can get oov
        cons: 1. slow
        """
        w2v = word2vec.load_txt_vec(filename, wordmap)
        for word in w2v:
            vector_size = len(w2v[word])
            break
        
        #embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        embedding_vectors = np.zeros((len(wordmap), vector_size))
        
        oov = []
        for i in range(len(wordmap)):
            word = wordmap[i]
            if (w2v.has_key(word)):
                embedding_vectors[i] = np.asarray(w2v[word], dtype="float32")
            else:
                oov.append(word)

        print ("OOV", oov)
        return embedding_vectors