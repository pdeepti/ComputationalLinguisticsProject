import os
import string
import subprocess
from subprocess import check_output,call
import operator
import math
import xml.etree.ElementTree as ET
import sys
sys.path.append("/home1/c/cis530/hw3/liblinear/python")
from liblinearutil import *
import pickle

 
def get_all_files_abs(directory):
    paths = []
    for path, directories, files in os.walk(directory):
        for filename in files:
            paths.append(os.path.join(path, filename))
    return paths

def preprocess(raw_text_file, core_nlp_output):
    call(["java","-cp",  "/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/stanford-corenlp-2012-07-09.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/stanford-corenlp-2012-07-06-models.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/xom.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/joda-time.jar", "-Xmx3g", "edu.stanford.nlp.pipeline.StanfordCoreNLP","-annotators", "tokenize,ssplit,pos,lemma,ner,parse", "-filelist", raw_text_file,"-outputDirectory",core_nlp_output])

def extract_top_words(xml_directory):
    top_words = []
    top_list = {}
    words = check_output(["grep", "<word>.*</word>", "-hor",xml_directory]).replace('<word>','').replace('</word>','').lower().split('\n')
    for w in words:
                if w in top_list:
                        top_list[w]+=1
                else:
                        top_list[w]=1
    sorted_words = map(lambda x:x[0],sorted(top_list.items(), key=operator.itemgetter(1),reverse=True))


    return sorted_words[:2000]

def split_data(xml_dir, label_txt):
	train_split = '/home1/a/anwesha/CL/project/train_split'
	test_split = '/home1/a/anwesha/CL/project/test_split'
	#files = get_all_files_abs(xml_dir)
	f = open(label_txt,'r')
	count = 0
	for line in f:
		name = line.split()[0]+'.xml'
		label = line.split()[1]
		if(count % 2 == 0):
			output_dir = train_split
		else:		
			output_dir = test_split
		cmd = ["cp", os.path.join(xml_dir, name), os.path.join(output_dir,'label_'+label+'_'+name)]
		subprocess.call(cmd)
		count+=1

def map_unigrams(xml_filename,top_words):
    word_list=[]
    i=0
    n= len(top_words)
    vector=[None]*n
    tree = ET.parse(xml_filename)
    root = tree.getroot()
    for token in root.iter('token'):
        word = token[0].text.lower()
        word_list.extend([word])

    for word in top_words:
        if word in word_list:
            vector[i]=1
        else:
            vector[i]=0
        i+=1
    return vector

def cosine_similarity(X,Y):
    numerator = 0
    X_square_sum = 0
    Y_square_sum = 0

    for i in range(0, len(X)):
        X_square_sum +=((float (X[i])) * (float (X[i]))*1.0)

        numerator += ((float(X[i])) * (float(Y[i]))*1.0)
    #print numerator
        Y_square_sum += ((float(Y[i])) *(float (Y[i]))*1.0)
    #print Y_square_sum
    denominator = math.sqrt(X_square_sum)*math.sqrt(Y_square_sum)
    if denominator == 0:
        return 0
    else:
        cosine = float(numerator)/denominator
    return cosine

def extract_similarity(top_words):
    f = open("/project/cis/nlp/tools/word2vec/vectors.txt",'r')
    vectors = f.read().split("\n")
    vec_dict = {}
    sim_mat = {}
    for lst in vectors:
        word_vec = []
        vec = lst.split()
        if len(vec) > 0:
                for i in range(1,len(vec)):
                        word_vec.append(vec[i])
                vec_dict[vec[0]] = word_vec
   # print vec_dict['the']
    for word1 in top_words:
        if word1 in vec_dict:
                similarity_dict = {}
                for word2 in top_words:
                        if word2 in vec_dict:
                                cosine_sim = cosine_similarity(vec_dict[word1],vec_dict[word2])
                                if cosine_sim != 0:
                                        similarity_dict[word2] = cosine_sim
                sim_mat[word1] = similarity_dict
    return sim_mat

def map_expanded_unigrams(xml_filename, top_words, similarity_matrix) :

    vec = map_unigrams(xml_filename, top_words)
    nonzero_words = []
    #print len(vec)
    #print len(top_words)
    for i in range(len(vec)):
        if vec[i] != 0:
                nonzero_words.append(top_words[i])

    for i in range(len(vec)):
        if vec[i] == 0:
                word = top_words[i]
                maximum = 0
                if word in similarity_matrix:
                        for word2 in nonzero_words:
                                if word2 in similarity_matrix:
                                        if similarity_matrix[word][word2] > maximum:
                                                maximum = similarity_matrix[word][word2]
                        vec[i] = maximum

    return vec

def extract_top_dependencies(xml_directory):
     top_dep =  []
     dict = {}
     paths = get_all_files_abs(xml_directory)
     for p in paths:
           tree = ET.parse(p)
           root = tree.getroot()
           for dep in root.iter('basic-dependencies'):
                for child in dep:
                        tuple = (child.attrib.get('type'),child[0].text.lower(),child[1].text.lower())
                        if tuple in dict:
                                dict[tuple]+=1
                        else:
                                dict[tuple]=1
     top_dep = map(lambda x:x[0],sorted(dict.items(), key=operator.itemgetter(1),reverse=True))             
     return top_dep[:2000]

def map_dependencies(xml_file, dep_list):
        vec = [0]*len(dep_list)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        list_tup = {}
        for dep in root.iter('basic-dependencies'):
                for child in dep:
                        tuple = (child.attrib.get('type'),child[0].text.lower(),child[1].text.lower())
                        list_tup[tuple] = 1
        for i in range(len(dep_list)):
                if dep_list[i] in list_tup:
                        vec[i]=1
        return vec

def map_prod_rules(xml_file, rule_list):

     vec = [0]*len(rule_list)
     top_prod = []
     top_prod = extract_prod_rules_from_file(xml_file)
     for i in range(len(rule_list)):
        if rule_list[i] in top_prod:
            vec[i]=1



     return vec


def extract_prod_rules(xml_directory):
    top_prod = []
    paths = get_all_files_abs(xml_directory)
    dict = {}
    for p in paths:
        rules = extract_prod_rules_from_file(p)
        for r in rules:
            if r in dict:
                dict[r]+=1
            else:
                dict[r]=1
    top_prod = map(lambda x:x[0],sorted(dict.items(), key=operator.itemgetter(1),reverse=True))
    return top_prod[:2000]

def extract_prod_rules_from_file(file_name):
    rule_list=[]
    split_rule=[]
    stack=[]
    rules_list=[]
    dict = {}
    tree = ET.parse(file_name)
    root = tree.getroot()
    for parse in root.iter('parse'):
        rule = parse.text
        rule_list.extend([rule])

    for rule in rule_list:
        split_rule = rule.split()
        for term in split_rule:
            if '(' in term:
                stack.append(term)
            else:
                no_para= term.count(')')
                if no_para == 1:
                    stack.append(term)
                else:
                    no_para-=1
                    temp_rule=''
                    previous_word=''
                    while stack and no_para !=0:
                       current_word = stack.pop()

                       if '(' in current_word:
                           temp_rule=current_word[1:]+'_'+temp_rule
                           if '(' in previous_word:
                                no_para -= 1
                                stack.append(current_word)
                                stack.append(')');
                                rules_list.extend([temp_rule[:-1]])
                                temp_rule=''

                       previous_word = current_word
    return rules_list







def process_corpus( xml_root, top_words, similarity_matrix, top_dependencies, syntactic_prod_rules ) :
    paths = get_all_files_abs(xml_root)
    if 'test' in xml_root:
        #f1 = open((os.path.join(os.getcwd(),"test_1.txt")),'w')
        #f2 = open((os.path.join(os.getcwd(),"test_2.txt")),'w')
        #f3 = open((os.path.join(os.getcwd(),"test_3.txt")),'w')
        #f4 = open((os.path.join(os.getcwd(),"test_4.txt")),'w')
        f5 = open((os.path.join(os.getcwd(),"test_5.txt")),'w')
    else:
        #f1 = open((os.path.join(os.getcwd(),"train_1.txt")),'w')
        #f2 = open((os.path.join(os.getcwd(),"train_2.txt")),'w')
        #f3 = open((os.path.join(os.getcwd(),"train_3.txt")),'w')
        #f4 = open((os.path.join(os.getcwd(),"train_4.txt")),'w')
        f5 = open((os.path.join(os.getcwd(),"train_5.txt")),'w')


    #vec1 = []
    for file in paths:
        vec1 = map_unigrams(file, top_words)
        vec2 = map_expanded_unigrams(file, top_words, similarity_matrix)
        vec3 = map_dependencies(file, top_dependencies)
        vec4 = map_prod_rules(file, syntactic_prod_rules)
        vec5 = vec1+vec3+vec4
        filename = os.path.basename(file).split('_')[1]
	if filename != '-1' and filename != '1':
		filename = '1'
        #str1 = filename
        #for i in range(len(vec1)):
        #        if vec1[i] != 0:
        #                k = i+1
        #                str1 = str1+' '+str(k)+':'+str(vec1[i])
        #f1.write(str1+'\n')
        #str2 = filename
        #for i in range(len(vec2)):
        #        if vec2[i] != 0:
        #                k = i+1
        #                str2 = str2+' '+str(k)+':'+str(vec2[i])
        #f2.write(str2+'\n')
        #str3 = filename
        #for i in range(len(vec3)):
        #        if vec3[i] != 0:
        #                k = i+1
        #                str3 = str3+' '+str(k)+':'+str(vec3[i])
        #f3.write(str3+'\n')
	#str4 = filename
        #for i in range(len(vec4)):
        #        if vec4[i] != 0:
        #                k = i+1
        #                str4 = str4+' '+str(k)+':'+str(vec4[i])
        #f4.write(str4+'\n')
        str5 = filename
        for i in range(len(vec5)):
                if vec5[i] != 0:
                        k = i+1
                        str5 = str5+' '+str(k)+':'+str(vec5[i])
        f5.write(str5+'\n')
   # f1.close()
   # f2.close()
   # f3.close()
   # f4.close()
    f5.close()



    return 0
 
def submission(test_dir):
	labels, acc, prob = run_classifier('train_5.txt','test_5.txt')
	f = open('results.txt','w')
	files = get_all_files_abs(test_dir)
	i = 0
	for filename in files:
		f.write(os.path.basename(filename)[:-4]+' '+str(int(labels[i]))+'\n')
		i+=1
	f.close()

def run_classifier(train_file, test_file):
    output_tuple = ()
    (Y_train, X_train) = svm_read_problem(train_file)
    v1 = (Y_train.count(-1))/float(len(Y_train))
    v2 = (Y_train.count(1))/float(len(Y_train))
    model = train(Y_train, X_train, "-s 0 -w1 "+ str(v1) + " -w-1 " + str(v2))
    (Y_test, X_test) = svm_read_problem(test_file)
    p_labels, p_acc, probs = predict(Y_test,X_test,model,"-b 1")
    return (p_labels, p_acc, probs)
    
if __name__ == "__main__":
	#root1 = "/mnt/castor/seas_home/c/cis530/hw3/data"
       	#train_data = "/home1/c/cis530/project/train_data"
	test_data = "/home1/c/cis530/project/test_data"
	#paths = get_all_files_abs(test_data)
	#testfiles_list = "/home1/a/anwesha/CL/project/trainfiles_list.txt"
	#fwrite = open(testfiles_list,'w')
	#for p in paths:
	#	fwrite.write(p+'\n')
	#fwrite.close()
	xml_train = "/home1/a/anwesha/CL/project/train_xml"
	xml_test = "/home1//a/anwesha/CL/project/xml_test"
	#labels = "/home1/a/anwesha/CL/project/train_labels.txt"
	#train_split = '/home1/a/anwesha/CL/project/train_split'
	#test_split = '/home1/a/anwesha/CL/project/test_split'
	#split_data(xml_train,labels)
	#preprocess(testfiles_list,xml_test)
	 # PICKLING
       #	top_words = extract_top_words(train_split)
       #	pickle.dump(top_words, open("top_words.p", "w")) 
       #	sim =  extract_similarity(top_words) 
       #	pickle.dump(sim, open("sim.p", "w")) 
       #	dep_list =  extract_top_dependencies(train_split)
       #	pickle.dump(dep_list,  open("dep_list.p", "w")) 
       #	rules = extract_prod_rules(train_split)
       #	pickle.dump(rules, open("rules.p", "w")) 
	 # UNPICKLING
       	#top_words = pickle.load(open("top_words.p", "r"))
       	#sim = pickle.load(open("sim.p", "r"))
       	#dep_list = pickle.load(open("dep_list.p", "r"))
       	#rules = pickle.load(open("rules.p","r"))
	#process_corpus(xml_train,top_words,sim,dep_list,rules)
	#process_corpus(xml_test, top_words, sim, dep_list, rules)
	#run_classifier('train_5.txt','test_5.txt')
	submission(xml_test)
