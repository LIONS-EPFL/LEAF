import numpy as np
import torch
import string
from time import time

from torchmetrics.multimodal.clip_score import CLIPScore
import nltk
nltk.download('punkt_tab')
nltk.download('words')
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from collections import OrderedDict
from copy import deepcopy
'''
Wrappers
------------------------------------------------------------------------------------------------------------------
'''


def convert_clip_text_model(model_with_projection, model_without_projection):
    # Assume:
    # model_with_projection: CLIPTextModelWithProjection
    # model_without_projection: CLIPTextModel

    # 1. Get state dicts
    state_dict_with_proj = model_with_projection.state_dict()
    state_dict_without_proj = model_without_projection.state_dict()

    # 2. Filter out the 'text_projection' weights from being overwritten
    #    and prepare a new OrderedDict that only updates `text_model.*`
    new_state_dict = OrderedDict()
    for name, param in state_dict_without_proj.items():
        # Prefix all keys with "text_model." to match in model_with_projection
        new_name = name
        if new_name in state_dict_with_proj:
            new_state_dict[new_name] = param

    # 3. Load the updated weights (non-strict to skip unmatched keys like text_projection)
    model_with_projection_2 = deepcopy(model_with_projection)
    missing_keys, unexpected_keys = model_with_projection_2.load_state_dict(new_state_dict, strict=False)

    # 4. Optional: print out what was skipped
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    return model_with_projection_2


def encode_text_wrapper(self, x, normalize = False):
    out = self(x).pooler_output
    if normalize:
        out = out / torch.norm(out,dim=-1,keepdim=True)
    return out

def encode_text_wrapper_CLIPModel(self, x, normalize = False):
    out = self.get_text_features(x)
    if normalize:
        out = out / torch.norm(out,dim=-1,keepdim=True)
    return out

class tokenizer_wrapper():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, x):
        return torch.tensor(self.tokenizer(x,padding=True,truncation=True).input_ids)



'''
Text related
------------------------------------------------------------------------------------------------------------------
'''

def valid_sentence(original,attacked,debug=False):
    '''
    Returns true or false based on the attacked sentence being a valid attack.
    An attack is valid if it doesn't introduce new words.
    attacked might be a list of attacked sentences
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10741578
    '''
    if isinstance(attacked, str):
        attacked = [attacked]
    if debug:
        start = time()
    W = set(words.words())
    if debug:
        end = time()
        print(f"Time to load words: {end - start}")
        start = time()
    # number of words in the original sentence
    #lo = len([w for w in word_tokenize(original.lower()) if w in W])
    lo = len(W.intersection(word_tokenize(original.lower())))
    if debug:
        end = time()
        print(f"Time to compute lo: {end - start}")
        start = time()
    # number of words in the attacked sentences
    #LA = [len([w for w in word_tokenize(a.lower()) if w in W]) for a in attacked]
    LA = [len(W.intersection(word_tokenize(a.lower()))) for a  in attacked]
    if debug:
        print(lo,LA)
    return [la < lo for la in LA]

def valid_sentence_batched(original,attacked,debug=False):
    '''
    Returns true or false based on the attacked sentence being a valid attack.
    An attack is valid if it doesn't introduce new words.
    attacked might be a list of attacked sentences
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10741578
    '''
    if isinstance(attacked, str):
        attacked = [[attacked]]
    if isinstance(attacked[0], str):
        attacked = [attacked]
    if isinstance(original, str):
        original = [original]
    if debug:
        start = time()
    W = set(words.words())
    if debug:
        end = time()
        print(f"Time to load words: {end - start}")
        start = time()
    # number of words in the original sentence
    #lo = len([w for w in word_tokenize(original.lower()) if w in W])
    lo = [len(W.intersection(word_tokenize(o.lower()))) for o in original]
    if debug:
        end = time()
        print(f"Time to compute lo: {end - start}")
        start = time()
    # number of words in the attacked sentences
    #LA = [len([w for w in word_tokenize(a.lower()) if w in W]) for a in attacked]
    LA = [[len(W.intersection(word_tokenize(a.lower()))) for a in AS] for AS in attacked]
    if debug:
        print(lo,LA)
    #return [[la < lo for la in LA] for lo in lo]
    return [[(l < lo) for l in la] for lo,la in zip(lo,LA)]

def margin_loss_lm(logits, true_class):
    '''
    Standard margin loss for classification
    '''
    #maximum different than true class
    max_other,_ = (torch.cat((logits[:,:true_class], logits[:,true_class+1:]), dim=-1)).max(dim=-1)
    return max_other - logits[:,true_class]

class margin_loss_lm_batched():
    def __init__(self,reduction = 'None'):
        self.reduction = reduction
    
    def __call__(self,logits, true_classes):
        '''
        Standard margin loss for classification
        '''
        L = torch.cat([margin_loss_lm(l.unsqueeze(0), t) for l,t in zip(logits,true_classes)], dim=0)
        if self.reduction == 'mean':
            return torch.mean(L)
        elif self.reduction == 'sum':
            return torch.sum(L)
        else:
            return L

def generate_sentence(S,z,u, V,k=1, alternative = None):
    '''
    inputs:
    S: sentence that we want to modify
    z: location position
    u: selection character id
    V: vocabulary, list of UNICODE indices
    k: number of possible changes
    
    generate sentence with a single character modification at position z with character u
    '''
    spaces = ''.join(['_' for i in range(k)])
    xx = ''.join([spaces + s for s in S] + [spaces])
    new_sentence = [c for c in xx]
    mask = []
    for i in range(len(S)):
        mask += [0 for i in range(k)] + [1]
    mask+=[0 for i in range(k)]
    
    if type(z) == list:
        for p,c in zip(z,u):
            if V[c] != -1:
                new_sentence[p] = chr(V[c])
                mask[p] = 1
            else: 
                new_sentence[p] = '_'
                mask[p] = 0
    else:
        if V[u] != -1:
            if new_sentence[z] == chr(V[u]) and (alternative is not None) and alternative != -1:
                new_sentence[z] = chr(alternative)
                mask[z] = 1
            elif new_sentence[z] == chr(V[u]) and (alternative is not None) and alternative == -1:
                new_sentence[z] = '_'
                mask[z] = 0
            else:
                new_sentence[z] = chr(V[u])
                mask[z] = 1
        else: 
            new_sentence[z] = '_'
            mask[z] = 0
    
    new_sentence = [c if mask[i] else '' for i,c in enumerate(new_sentence)]
    new_sentence = ''.join(new_sentence)
    return new_sentence

def generate_all_sentences_at_z(S, z, V,k=1, alternative = -1):
    '''
    inputs:
    S: sentence that we want to modify
    z: location id
    V: vocabulary, list of UNICODE indices
    
    generates all the possible sentences by changing characters in the position z
    '''
    return [generate_sentence(S,z,u, V,k, alternative=alternative) for u in range(len(V))]

def generate_random_sentences_at_z(S, z, V,n,k=1, alternative = -1):
    '''
    inputs:
    S: sentence that we want to modify
    z: location id
    V: vocabulary, list of UNICODE indices
    n: number of random samples
    
    generates all the possible sentences by changing characters in the position z
    '''
    return [generate_sentence(S,z,u, V,k, alternative=alternative) for u in np.random.choice(range(len(V)), size=n,replace = (n>len(V)))]

def generate_random_sentences(S,V,n,subset_z = None,k=1, alternative = None,insert=True):
    '''
    inputs:
    S: sentence that we want to modify
    V: vocabulary, list of UNICODE indices
    n: number of random samples to draw
    subset_z: subset of positions to consider
    k: number of character modifications
    alternative: in the case len(V)=1, character to consider for switchings when the character to change is
    the one in the volcabulary
    

    generates n random sentences at distance k
    '''
    if subset_z is None:
        subset_z = range(2*len(S) + 1)

    out = [S for _ in range(n)]

    for _ in range(k):
        
        if k==1:
            if not insert:
                subset_z =  [i for i in range(2*len(S)+1) if i%2]
            positions = np.random.choice(subset_z,size=n)
        else:
            if not insert:
                positions = [np.random.choice([i for i in range(2*len(s)+1) if i%2],size=1).item() for s in out]
            else:
                positions = [np.random.choice(range(2*len(s) + 1),size=1).item() for s in out]
        #print(positions)
        replacements = np.random.choice(range(len(V)),size=n)
        #print(out[0],positions[0],replacements[0])

        out = [generate_sentence(s,z,u, V,1, alternative=alternative) for s,z,u in zip(out,positions,replacements)]
    return out

def generate_all_sentences(S,V,subset_z = None,k=1, alternative = None):
    '''
    inputs:
    S: sentence that we want to modify
    V: vocabulary, list of UNICODE indices
    subset_z: subset of positions to consider
    k: number of character modifications (TODO: k>1)
    alternative: in the case len(V)=1, character to consider for switchings when the character to change is
    the one in the volcabulary
    
    generates all the possible sentences by changing characters
    '''
    out = []
    if subset_z is None:
        subset_z = range((k+1)*len(S) + k)
    for z in subset_z:
        out += generate_all_sentences_at_z(S, z, V, k, alternative=alternative)
    #return list(set(out)) #Avoid repeated sentences, this makes the algorithm not deterministic somehow
    #if subset_z != range((k+1)*len(S) + k):
    #    print(len(out), len(set(out)))
    return out

def attack_text_iterative(model,tokenizer,sentences,anchor_features,device,objective="l2",n=10,k=1, V=[-1] + [ord(c) for c in string.ascii_lowercase + ' ' + string.ascii_uppercase + string.digits + string.punctuation],debug=False):
    '''
    Attack the text

    model: clip model
    tokenizer: text tokenizer
    sentences: batch of clean sentences
    image_features: batch of encoded images with model
    n: number of random perturbations to sample for each sentence
    k: maximum Levenshtein distance to consider in the perturbations
    '''
    if objective == "dissim":
        '''
        just in case
        '''
        anchor_features /= anchor_features.norm(dim=-1, keepdim=True)

    for _ in range(k):
        SS = []
        for S in sentences:
            SS+= generate_random_sentences(S,V,n,k=1)
        tokens = tokenizer(SS).to(device)

        #print(tokens)
        if objective == 'l2':

            text_features = model.encode_text(tokens,normalize=False).view(len(sentences),n,-1)

            loss = ((text_features - anchor_features.view(len(sentences),1,-1))**2).sum(dim=-1)

            

        elif objective == 'dissim':
            text_features = model.encode_text(tokens,normalize=True).view(len(sentences),n,-1)

            loss = -(text_features @ anchor_features.transpose(-1,-2)).squeeze(-1)

        ids_best = torch.argmax(loss,dim=-1)

        sentences = []
        for row,id in enumerate(ids_best):
            sentences.append(SS[row*n + id])
        
    return torch.take_along_dim(text_features, ids_best.view(-1,1,1).repeat(1,1,text_features.shape[-1]),dim=1).squeeze(1), sentences

def attack_text_simple(model,tokenizer,sentences,anchor_features,device,objective="l2",n=10,k=1, V=[-1] + [ord(c) for c in string.ascii_lowercase + ' ' + string.ascii_uppercase + string.digits + string.punctuation],debug=False):
    '''
    Attack the text

    model: clip model
    tokenizer: text tokenizer
    sentences: batch of clean sentences
    image_features: batch of encoded images with model
    n: number of random perturbations to sample for each sentence
    k: maximum Levenshtein distance to consider in the perturbations
    '''

    if objective == "dissim":
        '''
        just in case
        '''
        anchor_features /= anchor_features.norm(dim=-1, keepdim=True)

    SS = []
    for S in sentences:
        SS+= generate_random_sentences(S,V,n,k=k)
    tokens = tokenizer(SS).to(device)

    #print(tokens)
    if objective == 'l2':

        text_features = model.encode_text(tokens,normalize=False).view(len(sentences),n,-1)

        loss = ((text_features - anchor_features.view(len(sentences),1,-1))**2).sum(dim=-1)

    elif objective == 'dissim':
        text_features = model.encode_text(tokens,normalize=True).view(len(sentences),n,-1)

        loss = -(text_features @ anchor_features.transpose(-1,-2)).squeeze(-1)

    ids_best = torch.argmax(loss,dim=-1)

    sentences = []
    for row,id in enumerate(ids_best):
        sentences.append(SS[row*n + id])

    return torch.take_along_dim(text_features, ids_best.view(-1,1,1).repeat(1,1,text_features.shape[-1]),dim=1).squeeze(1), sentences

def attack_text_charmer(model,tokenizer,sentences,anchor_features,device,objective="l2",n=10,k=1, V=[-1] + [ord(c) for c in string.ascii_lowercase + ' ' + string.ascii_uppercase + string.digits + string.punctuation], constrain = False,debug=False):
    '''
    n in this case is the number of random positions and random chars to replace.

    This attack is used for training as it can handle a batch of sentences in parallel.
    '''

    if objective in ["dissim","sim"]:
        '''
        just in case
        '''
        anchor_features /= anchor_features.norm(dim=-1, keepdim=True)

    for _ in range(k):
        #Select best positions from a random sample
        SS = []
        VV = [ord(' ')]
        positions = []

        for S in sentences:
            positions.append(np.random.choice(range(2*len(S)+1),size=n,replace = n>2*len(S)+1 ))
            SS.append(generate_all_sentences(S,VV,subset_z=positions[-1],alternative=-1))

        
        if constrain:
            valid = valid_sentence_batched(sentences,SS,debug=False)
            for i,S in enumerate(sentences):
                for j in range(n):
                    SS[i][j] = SS[i][j] if valid[i][j] else S
        # flatten the list of lists
        SS = [item for sublist in SS for item in sublist]
        tokens = tokenizer(SS).to(device)

        text_features = model.encode_text(tokens,normalize=(objective in ["sim","dissim"])).view(len(sentences),n,-1)

        if objective == 'l2':

            loss = ((text_features - anchor_features.view(len(sentences),1,-1))**2).sum(dim=-1)

        if objective == 'negl2':

            loss = -((text_features - anchor_features.view(len(sentences),1,-1))**2).sum(dim=-1)

        elif objective == 'dissim':

            loss = -(text_features @ anchor_features.transpose(-1,-2)).squeeze(-1)
        
        elif objective == 'sim':

            loss = (text_features @ anchor_features.transpose(-1,-2)).squeeze(-1)

        ids_best = torch.argmax(loss,dim=-1)

        del text_features, tokens
        best_pos = []
        for row,id in enumerate(ids_best):
            best_pos.append(positions[row][id])
        
        SS = []
        for i,S in enumerate(sentences):
            SS.append(generate_random_sentences_at_z(S, best_pos[i], V,n, alternative = -1))

        
        if constrain:
            valid = valid_sentence_batched(sentences,SS,debug=False)
            for i,S in enumerate(sentences):
                for j in range(n):
                    SS[i][j] = SS[i][j] if valid[i][j] else S
        # flatten the list of lists
        SS = [item for sublist in SS for item in sublist]
        tokens = tokenizer(SS).to(device)
        text_features = model.encode_text(tokens,normalize=(objective in ["sim","dissim"])).view(len(sentences),n,-1)

        if objective == 'l2':

            loss = ((text_features - anchor_features.view(len(sentences),1,-1))**2).sum(dim=-1)
        
        elif objective == 'negl2':

            loss = -((text_features - anchor_features.view(len(sentences),1,-1))**2).sum(dim=-1)

        elif objective == 'dissim':

            loss = -(text_features @ anchor_features.transpose(-1,-2)).squeeze(-1)

        elif objective == 'sim':

            loss = (text_features @ anchor_features.transpose(-1,-2)).squeeze(-1)
        
        ids_best = torch.argmax(loss,dim=-1)
        sentences = []
        for row,id in enumerate(ids_best):
            sentences.append(SS[row*n + id])
        if debug:
            print(sentences[0], torch.max(loss))

    return torch.take_along_dim(text_features, ids_best.view(-1,1,1).repeat(1,1,text_features.shape[-1]),dim=1).squeeze(1), sentences

def attack_text_bruteforce(model,tokenizer,sentence,anchor_features,device,batch_size=20*128,objective="l2",k=1, V=[-1] + [ord(c) for c in string.ascii_lowercase + ' ' + string.ascii_uppercase + string.digits + string.punctuation], constrain = False,debug=False):
    '''
    bruteforce for k=1
    '''
    if objective == "dissim":
        '''
        just in case
        '''
        anchor_features /= anchor_features.norm(dim=-1, keepdim=True)
    
    original = sentence

    with torch.no_grad():
        dist = 0
        
        #Generate all possible sentences with the top positions
        SS = generate_all_sentences(sentence,V,alternative=-1)
       
        # Only consider valid attacks according to the criterion of not creating new words
        if constrain:
            valid = valid_sentence_batched(original,SS,debug=False)
            valid = [item for sublist in valid for item in sublist]
            SS = [s if v else original for s,v in zip(SS,valid)]

        loss = []
        for i in range(len(SS)//batch_size + 1):
            beginning = i*batch_size
            end = min((i+1)*batch_size,len(SS)-1)
            
            if debug:
                print(beginning, end)
            if beginning==end:
                continue

            tokens = tokenizer(SS[beginning:end]).to(device)

            if objective == 'l2':
                text_features = model.encode_text(tokens,normalize=False).view(len(tokens),-1)

                l = ((text_features - anchor_features)**2).sum(dim=-1)

            elif objective == 'dissim':
                text_features = model.encode_text(tokens,normalize=True).view(len(tokens),-1)

                l = -(text_features @ anchor_features.transpose(-1,-2)).squeeze(-1)
            loss.append(l)

        loss = torch.cat(loss,dim=0)

        sentence = SS[torch.argmax(loss).item()]

        if debug:
            print(sentence, torch.max(loss))

    return sentence,dist+1

def attack_text_charmer_inference(model,tokenizer,sentence,anchor_features,device,objective="l2",n=10,k=1, V=[-1] + [ord(c) for c in string.ascii_lowercase + ' ' + string.ascii_uppercase + string.digits + string.punctuation], constrain = False,debug=False, batch_size=20*128, model_2 = None, model_2_anchor_features = None):
    '''
    n in this case is the number of positions in charmer

    THIS ATTACK CAN ONLY ATTACK 1 SENTENCE AT A TIME

    We use this attack during inference as it is the original charmer attack from https://arxiv.org/abs/2405.04346
    We don't use it during training as it is length-dependent and non parallelizable

    We assume that model_2 is another text encoder with the same tokenizer as model.
    '''
    if objective in ["dissim","sim"]:
        '''
        just in case
        '''
        anchor_features /= anchor_features.norm(dim=-1, keepdim=True)

        if model_2 is not None:
            model_2_anchor_features /= model_2_anchor_features.norm(dim=-1, keepdim=True)
    
    original = sentence

    with torch.no_grad():
        dist = 0
        for dist in range(k):
            #Select best positions
            VV = [ord(' ')]
            SS = generate_all_sentences(sentence,VV,alternative=-1)
            if constrain:
                valid = valid_sentence_batched(sentence,SS,debug=False)
                valid = [item for sublist in valid for item in sublist]
                SS = [s if v else sentence for s,v in zip(SS,valid) ]
    
            tokens = tokenizer(SS).to(device)
            loss = []
            for i in range(len(tokens)//batch_size +1):
                beginning = i*batch_size
                end = min((i+1)*batch_size,len(tokens)-1)
                if beginning >= end:
                    continue
                text_features = model.encode_text(tokens[beginning:end],normalize=(objective in ['sim', 'dissim'])).view(len(tokens[beginning:end]),-1)
                if model_2 is not None:
                    text_features_2 = model_2.encode_text(tokens[beginning:end],normalize=(objective in ['sim', 'dissim'])).view(len(tokens[beginning:end]),-1)
                if objective == 'l2':
                    if model_2 is not None:
                        loss.append(((text_features - anchor_features)**2).sum(dim=-1) + ((text_features_2 - model_2_anchor_features)**2).sum(dim=-1))/2
                    else:
                        loss.append(((text_features - anchor_features)**2).sum(dim=-1))

                elif objective == 'negl2':
                    if model_2 is not None:
                        loss.append(-(((text_features - anchor_features)**2).sum(dim=-1) + ((text_features_2 - model_2_anchor_features)**2).sum(dim=-1))/2)
                    else:
                        loss.append(-((text_features - anchor_features)**2).sum(dim=-1))

                elif objective == 'dissim':
                    if model_2 is not None:
                        loss.append(-((text_features @ anchor_features.transpose(-1,-2)).squeeze(-1) + (text_features_2 @ model_2_anchor_features.transpose(-1,-2)).squeeze(-1))/2)
                    else:
                        loss.append(-(text_features @ anchor_features.transpose(-1,-2)).squeeze(-1))
                
                elif objective == 'sim':
                    if model_2 is not None:
                        loss.append(((text_features @ anchor_features.transpose(-1,-2)).squeeze(-1) + (text_features_2 @ model_2_anchor_features.transpose(-1,-2)).squeeze(-1))/2)
                    else:
                        loss.append((text_features @ anchor_features.transpose(-1,-2)).squeeze(-1))
            loss = torch.cat(loss,dim=0)

            top_positions = torch.topk(loss,min(n,loss.shape[0]),dim=0).indices
            
            del text_features, tokens, loss
            
            #Generate all possible sentences with the top positions
            SS = generate_all_sentences(sentence,V,subset_z=top_positions,alternative=-1)

            # Only consider valid attacks according to the criterion of not creating new words
            # We compare against sentence, because maybe with one perturbation we reduce the number of words,
            # But then in the next we introduce a new one:
            # A big burly grizzly bear is show with grass in the background.
            # A big burly grizzly bear is show with grads in the background.
            # A big burly grizzly beer is show with grads in the background.
            if constrain:
                valid = valid_sentence_batched(sentence,SS,debug=False)
                valid = [item for sublist in valid for item in sublist]
                SS = [s if v else sentence for s,v in zip(SS,valid)]
                if len(SS) == 0:
                    SS = [sentence] 

            tokens = tokenizer(SS).to(device)
            loss = []
            for i in range(len(tokens)//batch_size + 1):
                beginning = i*batch_size
                end = min((i+1)*batch_size,len(tokens)-1)
                if beginning >= end:
                    continue
                text_features = model.encode_text(tokens[beginning:end],normalize=(objective in ["sim", "dissim"])).view(len(tokens[beginning:end]),-1)
                if model_2 is not None:
                    text_features_2 = model_2.encode_text(tokens[beginning:end],normalize=(objective in ["sim", "dissim"])).view(len(tokens[beginning:end]),-1)

                if objective == 'l2':   
                    if model_2 is not None: 
                        loss.append((((text_features - anchor_features)**2).sum(dim=-1) + ((text_features_2 - model_2_anchor_features)**2).sum(dim=-1))/2)
                    else:
                        loss.append(((text_features - anchor_features)**2).sum(dim=-1))

                elif objective == 'negl2':
                    if model_2 is not None:
                        loss.append(-(((text_features - anchor_features)**2).sum(dim=-1) + ((text_features_2 - model_2_anchor_features)**2).sum(dim=-1))/2)
                    else:
                        loss.append(-((text_features - anchor_features)**2).sum(dim=-1))

                elif objective == 'dissim':
                    if model_2 is not None:
                        loss.append(-((text_features @ anchor_features.transpose(-1,-2)).squeeze(-1) + (text_features_2 @ model_2_anchor_features.transpose(-1,-2)).squeeze(-1))/2)
                    else:
                        loss.append(-(text_features @ anchor_features.transpose(-1,-2)).squeeze(-1))

                elif objective == 'sim':
                    if model_2 is not None:
                        loss.append(((text_features @ anchor_features.transpose(-1,-2)).squeeze(-1) + (text_features_2 @ model_2_anchor_features.transpose(-1,-2)).squeeze(-1))/2)
                    else:
                        loss.append((text_features @ anchor_features.transpose(-1,-2)).squeeze(-1))
            loss = torch.cat(loss,dim=0)

            sentence = SS[torch.argmax(loss).item()]

            if debug:
                print(sentence, torch.max(loss))

        return sentence,dist+1

def attack_text_random_search(model,tokenizer,sentence,anchor_features,device,objective="l2",n=10,k_step = 1,k=1, V=[-1] + [ord(c) for c in string.ascii_lowercase + ' ' + string.ascii_uppercase + string.digits + string.punctuation], constrain = False,debug=False):
    '''
    random search
    '''
    if objective in ["dissim","sim"]:
        '''
        just in case
        '''
        anchor_features /= anchor_features.norm(dim=-1, keepdim=True)
    
    original = sentence

    with torch.no_grad():
        dist = 0
        for dist in range(k):
            #avoid inserts
            SS = generate_random_sentences(sentence,V,n,k=k_step,alternative=None,insert=False)
            # if debug:
            #     print(max([len(s) for s in SS]))
            tokens = tokenizer(SS).to(device)
            text_features = model.encode_text(tokens,normalize=(objective in ["sim", "dissim"])).view(len(SS),-1)

            if objective == 'l2':

                loss = ((text_features - anchor_features)**2).sum(dim=-1)

            elif objective == 'negl2':

                loss = -((text_features - anchor_features)**2).sum(dim=-1)

            elif objective == 'dissim':

                loss = -(text_features @ anchor_features.transpose(-1,-2)).squeeze(-1)

            elif objective == 'sim':

                loss = -(text_features @ anchor_features.transpose(-1,-2)).squeeze(-1)

            sentence = SS[torch.argmax(loss).item()]

            if debug:
                print(sentence, torch.max(loss))

        return sentence,dist+1

def attack_text_charmer_classification(model,tokenizer,sentence,image_features,label,device,n=10,k=1, V=[-1] + [ord(c) for c in string.ascii_lowercase + ' ' + string.ascii_uppercase + string.digits + string.punctuation],debug=False, batch_size=128*20):
    '''
    n in this case is the number of positions in charmer

    THIS ATTACK CAN ONLY ATTACK 1 SENTENCE AT A TIME
    '''
    criterion = margin_loss_lm_batched(reduction='None')

    with torch.no_grad():
        dist = 0
        for dist in range(k):
            #Select best positions
            VV = [ord(' ')]
            SS = generate_all_sentences(sentence,VV,alternative=-1)
            tokens = tokenizer(SS).to(device)
            loss = []
            for i in range(len(tokens)//batch_size + 1):
                beginning = i*batch_size
                end = min((i+1)*batch_size,len(tokens)-1)
                if beginning == end:
                    continue
                text_features = model.encode_text(tokens[beginning:end],normalize=True).view(len(tokens[beginning:end]),-1)

                text_sims = (text_features @ image_features.view(len(image_features),-1).transpose(0,1))

                loss.append(criterion(text_sims,label*torch.ones(len(SS),device=device).long().to(device)))
            
            loss = torch.cat(loss,dim=0)

            top_positions = torch.topk(loss,min(n,text_sims.shape[0]),dim=0).indices
            
            del text_features, tokens, text_sims
            
            #Generate all possible sentences with the top positions
            SS = generate_all_sentences(sentence,V,subset_z=top_positions,alternative=-1)
            tokens = tokenizer(SS).to(device)
            loss = []
            preds = []
            for i in range(len(tokens)//batch_size + 1):
                beginning = i*batch_size
                end = min((i+1)*batch_size,len(tokens)-1)
                if beginning == end:
                    continue
                text_features = model.encode_text(tokens[beginning:end],normalize=True).view(len(tokens[beginning:end]),-1)

                text_sims = (text_features @ image_features.view(len(image_features),-1).transpose(0,1))
                loss.append(criterion(text_sims,label*torch.ones(len(SS),device=device).long().to(device)))
                
                text_probs = text_sims.softmax(dim=-1)
                preds.append(text_probs.argmax(dim=-1))


            loss = torch.cat(loss,dim=0)
            preds = torch.cat(preds,dim=0)
            if debug:
                print(preds[:5], preds.shape)

            sentence = SS[torch.argmax(loss).item()]

            if preds[torch.argmax(loss).item()] != label:
                break
        return sentence,dist+1


def attack_text(model,tokenizer,sentences,image_features,device,objective="l2",n=10,k=1, V=[-1] + [ord(c) for c in string.ascii_lowercase + ' ' + string.ascii_uppercase + string.digits + string.punctuation],constrain = False,debug=False):
    #return attack_text_simple(model,tokenizer,sentences,image_features,device,n,k,V,debug)
    #return attack_text_iterative(model,tokenizer,sentences,image_features,device,n,k,V,debug)
    return attack_text_charmer(model,tokenizer,sentences,image_features,device,objective,n,k,V,constrain,debug)


'''
Image related
'''

def attack_image(model,normalize,images,anchor_features,device,objective="l2",eps=2/255,n_steps = 10, stepsize = None, debug=False):
    '''
    Attack the image

    model: clip model
    normalize: normalization transform for the images
    images: batch of images to attack
    anchor_features: batch of embeddings that we want to deviate from
    objective: objective function to maximize (l2 or dissim).
    eps: maximum infinite norm of the perturbations
    n_steps: number of PGD steps with stepsize eps/n_steps if none is given
    stepsize: PGD stepsize
    '''
    if stepsize is None:
        stepsize = eps/n_steps

    '''
    get over the dataparallel errors
    '''
    if model.__class__.__name__ == 'DistributedDataParallel':
        #print('hey')
        model = model.module

    if objective == "dissim":
        anchor_features /= anchor_features.norm(dim=-1, keepdim=True)

    delta = eps*(2*torch.rand(images.shape,device=device) - 1)
    for _ in range(n_steps):
        delta.requires_grad = True

        if objective == "l2":
            image_features = model.encode_image(image = normalize(images + delta),normalize = False).view(len(images),-1)
            loss = ((anchor_features - image_features)**2).sum()

        elif objective == "dissim":
            image_features = model.encode_image(image = normalize(images + delta),normalize = True).view(len(images),-1)
            loss = - (anchor_features*image_features).sum()

        loss.backward(retain_graph=True)
        delta = (delta + stepsize*torch.sign(delta.grad)).detach()
        delta = torch.clamp(delta,-eps,eps)
        if debug:
            print('loss:',loss.item())
    return (images + delta).detach()

def attack_image_classification(model,normalize,images,text_features,labels,device,eps=2/255,n_steps = 10, stepsize = None, debug=False):
    '''
    Attack the image in the classification setting

    model: clip model
    normalize: normalization transform for the images
    images: batch of images to attack
    text_features: batch of encoded sentences with model, one per class
    eps: maximum infinite norm of the perturbations
    n_steps: number of PGD steps with stepsize eps/n_steps if none is given
    stepsize: PGD stepsize
    '''
    if stepsize is None:
        stepsize = eps/n_steps

    criterion = torch.nn.CrossEntropyLoss()

    delta = eps*(2*torch.rand(images.shape,device=device) - 1)
    for _ in range(n_steps):
        delta.requires_grad = True
        #model.zero_grad()
        image_features = model.encode_image(normalize(images + delta)).view(len(images),-1)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ text_features
        
        loss = criterion(logits,labels)
        loss.backward(retain_graph=True)
        delta = (delta + stepsize*torch.sign(delta.grad)).detach()
        delta = torch.clamp(delta,-eps,eps)
        if debug:
            print('loss:',logits.softmax(-1).sum().item())
    return (images + delta).detach()


def attack_text_charmer_t2i_pipeline(pipeline,clip,clip_processor,sentence,device,batch_size=20,n=10,k=1, V=[-1] + [ord(c) for c in string.ascii_lowercase + ' ' + string.ascii_uppercase + string.digits + string.punctuation],debug=False):
    '''
    attacking the text to minimize the clip_score between the generated image and the original caption.
    '''
    text_inputs = clip_processor(text=[sentence], return_tensors="pt", padding=True).to(device)
    text_emb = clip.get_text_features(**text_inputs)
    text_emb/=torch.norm(text_emb,p=2,dim=-1,keepdim=True)

    with torch.no_grad():
        dist = 0
        for dist in range(k):
            #Select best positions
            VV = [ord(' ')]
            SS = generate_all_sentences(sentence,VV,alternative=-1)

            loss = []

            print(len(SS), len(SS))

            for i in range(len(SS)//batch_size):
                beginning = i*batch_size
                end = min((i+1)*batch_size,len(SS)-1)
                if debug:
                    print(beginning, end)
                out = pipeline(SS[beginning:end], num_inference_steps = 50).images

                images = clip_processor(images=out, return_tensors="pt").to(device)
                image_emb = clip.get_image_features(**images)

                image_emb/=torch.norm(image_emb,p=2,dim=-1,keepdim=True)

                loss.append(-image_emb@text_emb.T)
                if debug:
                    print(beginning, end,i,len(SS)//batch_size + 1,loss[-1])

            loss = torch.cat(loss,dim=0).squeeze()

            top_positions = torch.topk(loss,min(n,loss.shape[0]),dim=0).indices
            
            loss = []
            
            #Generate all possible sentences with the top positions
            SS = generate_all_sentences(sentence,V,subset_z=top_positions,alternative=-1)
            for i in range(len(SS)//batch_size):
                beginning = i*batch_size
                end = min((i+1)*batch_size,len(SS)-1)
                if debug:
                    print(beginning, end)
                out = pipeline(SS[beginning:end], num_inference_steps = 50).images

                images = clip_processor(images=out, return_tensors="pt").to(device)
                image_emb = clip.get_image_features(**images)

                image_emb/=torch.norm(image_emb,p=2,dim=-1,keepdim=True)

                loss.append(-image_emb@text_emb.T)
                if debug:
                    print(i,len(SS)//batch_size + 1,loss[-1])

            loss = torch.cat(loss,dim=0).squeeze()

            sentence = SS[torch.argmax(loss).item()]

        return sentence,dist+1

def attack_text_charmer_t2i_pipeline_fast(pipeline,clip,clip_processor,sentence,device,batch_size=20,n=10,k=1, V=[-1] + [ord(c) for c in string.ascii_lowercase + ' ' + string.ascii_uppercase + string.digits + string.punctuation],debug=False):
    '''
    attacking the text to minimize the clip_score between the generated image and the original caption.
    '''
    text_inputs = clip_processor(text=[sentence], return_tensors="pt", padding=True).to(device)
    text_emb = clip.get_text_features(**text_inputs)
    text_emb/=torch.norm(text_emb,p=2,dim=-1,keepdim=True)

    with torch.no_grad():
        dist = 0
        for dist in range(k):
            #Select best positions
            VV = [ord(' ')]
            positions = np.random.choice(range(2*len(sentence)+1),size=n,replace = n>2*len(sentence)+1 )
            SS = generate_all_sentences(sentence,VV,subset_z=positions,alternative=-1)
            
            loss=[]
            for i in range(len(SS)//batch_size):
                beginning = i*batch_size
                end = min((i+1)*batch_size,len(SS)-1)
                if debug:
                    print(beginning, end)
                out = pipeline(SS[beginning:end], num_inference_steps = 50).images

                images = clip_processor(images=out, return_tensors="pt").to(device)
                image_emb = clip.get_image_features(**images)

                image_emb/=torch.norm(image_emb,p=2,dim=-1,keepdim=True)

                loss.append(-image_emb@text_emb.T)
                if debug:
                    print(beginning, end,i,len(SS)//batch_size + 1,loss[-1])

            loss = torch.cat(loss,dim=0).squeeze()

            best_pos = torch.argmax(loss).item()
            
            SS= generate_random_sentences_at_z(sentence, best_pos, V,n, alternative = -1)
            
            loss = []
            for i in range(len(SS)//batch_size):
                beginning = i*batch_size
                end = min((i+1)*batch_size,len(SS)-1)
                if debug:
                    print(beginning, end)
                out = pipeline(SS[beginning:end], num_inference_steps = 50).images

                images = clip_processor(images=out, return_tensors="pt").to(device)
                image_emb = clip.get_image_features(**images)

                image_emb/=torch.norm(image_emb,p=2,dim=-1,keepdim=True)

                loss.append(-image_emb@text_emb.T)
                if debug:
                    print(i,len(SS)//batch_size + 1,loss[-1])

            loss = torch.cat(loss,dim=0).squeeze()

            sentence = SS[torch.argmax(loss).item()]

        return sentence,dist+1

if __name__ == '__main__':
    pass