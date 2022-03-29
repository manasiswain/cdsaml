from textblob import TextBlob
import pandas as pd
import demoji
import emoji as em
import nltk
import time
import regex as re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from stop_words import get_stop_words
from nltk.corpus import stopwords
nltk.download('wordnet')
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
# Create WordNetLemmatizer object
wnl = WordNetLemmatizer()
nltk.download('stopwords')
tptnfpfn={}
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
import wordninja
def deEmojify(inputString):
    sent_0 = re.sub('[^A-Za-z0-9]+', ' ', inputString)
    return sent_0.encode('ascii', 'ignore').decode('ascii')
def countpol(emojispol):
    pos=0
    neg=0
    for i in emojispol:
        if(i>0):
            pos+=1
        elif(i<0):
            neg+=1
    if(pos>neg):
        return(1)
    elif(pos<neg ):
        return(-1)
    elif(pos==neg):
        x=average(emojispol)
        return(x)

def average(emojispol):
    sum=0
    for i in emojispol:
        sum+=i
    return(sum/len(emojispol))

def checkans(ans,ourans,tp,tn,fp,fn):
    for i in range(0,len(ans)):
        if(ans[i]==ourans[i] and ans[i]==1):
            tp+=1
        elif (ans[i] == ourans[i] and ans[i] == 0):
            tn+=1
        elif(ourans[i]==1 and ans[i]==0):
            fp+=1
        elif(ourans[i] == 0 and ans[i] == 1):
            fn+=1
    tptnfpfn['TP'] = tp
    tptnfpfn['TN'] = tn
    tptnfpfn['FP'] = fp
    tptnfpfn['FN'] = fn
    ansct=tp+tn #accuracy
    t=[]
    accuracy=(ansct/len(ans))*100
    t.append(accuracy)
    precision=tp/(tp+fp)
    t.append(precision)
    recall=tp/(tp+fn)
    t.append(recall)
    specificity=tn/(tn+fp)
    t.append(specificity)
    f1=(2*recall*precision)/(recall+precision)
    t.append(f1)
    return(t)
def preprocess():
    # remove emoji
    for i in l:
        k.append(demoji.findall(i))
    b = []
    ct = 0
    o = []
    for i in l:
        o.append(deEmojify(i))
    # stop words
    new1 = []
    for i in o:
        h = []
        oo = i.split()
        for j in oo:
            if (j.casefold() not in stop):
                h.append(j)
        new1.append(h)

def our_model(X_train, X_test, y_train, y_test):
    ctt = 0
    '''l = []
    ll = []
    for i in X_test:
        ll.append(i)
        l.append(extract_emojis(i))'''
    counter = 0
    ourans = []
    countnew = 0
    for i in l:
        print(i)
        j = deEmojify(i)
        k = wordninja.split(j)
        j = " ".join(wordninja.split(j))
        j = wnl.lemmatize(j)
        testimonial = TextBlob(j)
        countnew += 1
        if (len(emojiscore[ctt]) > 0):
            x = countpol(emojiscore[ctt])

        else:
            x = 0
        ischeck2=0
        jj=i.split(' ')
        for p in jj:
            p=p.casefold()
            if(p.find("#Sarcastic".casefold())!=-1 or p.find("#Sarcastictweets".casefold())!=-1 or p.find("#Sarcasm".casefold())!=-1 or p.find("#Sarcasmo".casefold())!=-1 or  p.find("#irony".casefold())!=-1):
                ischeck2=1
                ourans.append(1)
                break
        if(ischeck2 == 0):
            ckk=0
            for jj in give():
                if(jj in i):
                    ckk=1
                    ourans.append(1)
                    break
            if(ckk==0):
                if ((testimonial.sentiment.polarity > 0 and x < 0) or (testimonial.sentiment.polarity < 0 and x > 0) or (
                    testimonial.sentiment.polarity == 0 and x < 0) or (testimonial.sentiment.polarity == 0 and x > 0)):
                # print(i,testimonial.sentiment.polarity,emojiscore[ctt],"SARCASTIC")
                    ourans.append(1)
                else:
                    ttt=0
                    for q in k:
                        test = TextBlob(q)
                        if ((testimonial.sentiment.polarity > 0 and test.sentiment.polarity < 0) or (
                            testimonial.sentiment.polarity < 0 and test.sentiment.polarity > 0)):
                            ourans.append(1)
                            ttt=1
                            break
                    if(ttt==0):
                        ourans.append(0)

        ctt += 1
    print(len(ans),len(ourans))
    acc(ans,ourans)
def acc(ans,ourans):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    ANS = checkans(ans,ourans,tp,tn,fp,fn)
    print("Accuracy:", ANS[0])
    print("Precision:", ANS[1] * 100)
    print("Recall:", ANS[2] * 100)
    print("Specificity:", ANS[3] * 100)
    print("F1:", ANS[4] * 100)

def our_model2(X_train, X_test, y_train, y_test,var):
   l=[]
   ll=[]
   for i in X_test:
       ll.append(i)
       l.append(extract_emojis(i))
   oa=[]
   ctt=0
   bigct=0
   for i in l:
       if(i!=''):
           j = deEmojify(ll[bigct])
           k = wordninja.split(j)
           j = " ".join(wordninja.split(j))
           j = wnl.lemmatize(j)
           testimonial = TextBlob(j)
           test = TextBlob(ll[bigct])
           ischeck2 = 0
           if (len(emojiscore[ctt]) > 0):
               x = countpol(emojiscore[ctt])
           else:
               x = 0
           ll[bigct] = ll[bigct].lower()
           if (ll[bigct].find("#sarcastic".lower()) != -1 or ll[bigct].find(
                   "#sarcastictweet".lower()) != -1 or ll[bigct].find("#sarcasm".lower()) != -1 or ll[
               bigct].find("#sarcasmo".lower()) != -1 or ll[bigct].find("#irony".lower()) != -1):
               ischeck2 = 1
               oa.append(1)
           if (ischeck2 == 0):
                r=nb_op(i,var)
                if(r==1):
                    oa.append(1)
                elif(r==2):
                    if(var==0):
                        np.delete(y_test,bigct)
                        continue
                    ckk = 0
                    for jj in give():
                        if (jj in ll[bigct]):
                            ckk = 1
                            oa.append(1)
                            break
                    if (ckk == 0):
                        if ((testimonial.sentiment.polarity > 0 and x < 0) or (
                                testimonial.sentiment.polarity < 0 and x > 0) or (
                                testimonial.sentiment.polarity == 0 and x < 0) or (
                                testimonial.sentiment.polarity == 0 and x > 0)):
                            oa.append(1)
                        else:
                            ttt=0
                            for q in k:
                                test = TextBlob(q)
                                if ((testimonial.sentiment.polarity > 0 and test.sentiment.polarity < 0) or (
                                        testimonial.sentiment.polarity < 0 and test.sentiment.polarity > 0)):
                                    oa.append(1)
                                    ttt=1
                                    break
                            if(ttt==0):
                                oa.append(0)

                else:
                    oa.append(0)


       else:
           j = deEmojify(ll[bigct])
           k = wordninja.split(j)
           j = " ".join(wordninja.split(j))
           j = wnl.lemmatize(j)
           testimonial = TextBlob(j)
           test = TextBlob(ll[bigct])
           ischeck2 = 0
           ll[bigct] = ll[bigct].lower()
           if (ll[bigct].find("#sarcastic".lower()) != -1 or ll[bigct].find("#sarcastictweet".lower()) != -1 or ll[bigct].find("#sarcasm".lower()) != -1 or ll[bigct].find("#sarcasmo".lower()) != -1 or ll[bigct].find("#irony".lower()) != -1):
               ischeck2=1
               oa.append(1)
           if(ischeck2==0):
                if ((testimonial.sentiment.polarity > 0 and test.sentiment.polarity < 0) or (
                   testimonial.sentiment.polarity < 0 and test.sentiment.polarity > 0)):
                    oa.append(1)
                else:
                    oa.append(0)
       ctt+=1
       bigct+=1
   acc(list(y_test),oa)
   return(oa)



def get_score(x):
    count=0
    for i in emoji:
        if i!=x:
            count=count+1
        else:
            q=sentiment_score[count]
            return q

    return 0
data = pd.read_csv('Facebook_Twitter.csv')
ddd=list(data)
emojiData=pd.read_csv("ijstable.csv")
fc=emojiData[emojiData.columns[0]]
emoji=emojiData["Char"]
emoji=list(emoji)

emoji.pop(0)

sentiment_score=emojiData["Sentiment score"]
sentiment_score=list(sentiment_score)
sentiment_score.pop(0)
sentiment_score= [float(i) for i in sentiment_score]
stop = list(get_stop_words('en'))#About 900 stopwords
nltk_words = list(stopwords.words('english')) #About 150 stopwords
stop.extend(nltk_words)
#sentiment=SentimentIntensityAnalyzer()
first_column = data[data.columns[0]]
second_column= data[data.columns[1]]
#TRAIN
X=first_column
y=second_column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=42)
l=list(first_column)
ans=list(second_column)
ans_train=X_train
k=[]
preprocess()
emojiscore=[]
for i in k:
    q=[]
    for j in i:
        q.append(get_score(j))
    emojiscore.append(q)
ct=0
def extract_emojis(s):
  return ''.join(c for c in s if c in em.UNICODE_EMOJI['en'])
d={}
dd={}
q=[]
q1=[]
ctsarcastic=0
ctnotsarcastic=0
for i in ans:
    if(i==1):
        ctsarcastic+=1
        qq=list(extract_emojis(l[ct]))
        for j in qq:
            if j not in q:
                q.append(j)
                d[j]=0
            else:
                d[j]+=1
    else:
        ctnotsarcastic+=1
        qq=list(extract_emojis(l[ct]))
        for j in qq:
            if j not in q1:
                q1.append(j)
                dd[j]=0
            else:
                dd[j]+=1
    ct+=1
psarcastic=ctsarcastic/(ctsarcastic+ctnotsarcastic) #for nb
pnotsarcastic=ctnotsarcastic/(ctsarcastic+ctnotsarcastic) #for nb
m=sorted(d.items(),key=lambda x:x[1],reverse=True)
mm=sorted(dd.items(),key=lambda x:x[1],reverse=True)
m=m[:]
mm=mm[:]
k=m[:10]
kk=mm[:10]
p={}
pp={}
d={}
dd={}
for i in m:
    d[i[0]]=i[1]
for i in mm:
    dd[i[0]]=i[1]
for i in k:
    p[i[0]]=i[1]
for i in kk:
    pp[i[0]]=i[1]
def give():
    l=[]
    b=[]
    for i in d:
        l.append([i,d[i]])
    l.sort(key=lambda x:x[1])
    l.reverse()
    ll = []
    c=[]
    bb=[]
    for i in dd:
        ll.append([i, dd[i]])
    ll.sort(key=lambda x: x[1])
    ll.reverse()
    #print(l)
    for i in range(0,7):
        b.append(l[i][0])
    for i in ll:
        bb.append(i[0])
    for i in b:
        if(i not in bb):
            c.append(i)
    return(c)
def nb():
    l = []
    b = []
    s1=0
    s2=0
    for i in d:
        s1+=d[i]
        l.append([i, d[i]])
    l.sort(key=lambda x: x[1])
    l.reverse()
    ll = []
    c = []
    bb = []
    for i in dd:
        s2+=dd[i]
        ll.append([i, dd[i]])
    ll.sort(key=lambda x: x[1])
    ll.reverse()
    for i in d:
        d[i]=(d[i]+1)/(s1+len(d))
    for i in dd:
        dd[i]=(dd[i]+1)/(s2+len(dd))

def nb_op(x,var):
    if(var==1):
        return(2)
    ps=psarcastic
    pns=pnotsarcastic
    for i in x:
        if(i in d.keys()):
            ps*=d[i]
        if (i in dd.keys()):
            pns *= dd[i]
    if(ps>pns):
        return(1)
    elif(ps<pns):
        return(0)
    else:
        return(2)


s1=time.time()
start=time.time()
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf,y_train)
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
text_clf = text_clf.fit(X_train,y_train)
predicted = text_clf.predict(X_test)
print("MNB")
print(np.mean(predicted == y_test))
end=time.time()
print("The time of execution of above program is :",end-start)
print("Our Model")
start=time.time()
#our_model2(X_train, X_test, y_train, y_test)
oa_our_model2=our_model2(X_train, X_test, y_train, y_test,1)
end=time.time()
print("The time of execution of above program is :", end-start)
names1 = list(p.keys())
values1 = list(p.values())
plt.subplot(2,1,1)
plt.title("TOP USED EMOJIS IN SARCASTIC SENTENCES",fontdict = {'fontsize':6 })
plt.xlabel("EMOJI",fontdict = {'fontsize':4})
plt.ylabel("FREQUENCY",fontdict = {'fontsize':6 })
plt.bar(range(len(p)), values1, tick_label=names1)
plt.subplot(2,1, 2) # index 2
names2 = list(tptnfpfn.keys())
values2 = list(tptnfpfn.values())
plt.ylabel("FREQUENCY",fontdict = {'fontsize':6 })
plt.bar(range(len(tptnfpfn)), values2, tick_label=names2)
plt.show()
print("Naive Bayes")
start=time.time()
nb()
#our_model2(X_train, X_test, y_train, y_test)
oa_nb=our_model2(X_train, X_test, y_train, y_test,0)
end=time.time()
print("The time of execution of above program is :", end-start)



print("Ensemble")
oa_final=[]
for i in range(0,len(oa_nb)):
    z=[oa_our_model2[i],oa_nb[i],predicted[i]]
    if(z.count(1)>1):
        oa_final.append(1)
    else:
        oa_final.append(0)
acc(list(y_test),oa_final)
e1=time.time()
print("The time of execution of above program is :", e1-s1)
'''
names1 = list(p.keys())
values1 = list(p.values())
plt.subplot(2,1,1)
plt.title("TOP USED EMOJIS IN SARCASTIC SENTENCES",fontdict = {'fontsize':6 })
plt.xlabel("EMOJI",fontdict = {'fontsize':4})
plt.ylabel("FREQUENCY",fontdict = {'fontsize':6 })
plt.bar(range(len(p)), values1, tick_label=names1)
plt.subplot(2,1, 2) # index 2
names2 = list(tptnfpfn.keys())
values2 = list(tptnfpfn.values())
plt.ylabel("FREQUENCY",fontdict = {'fontsize':6 })
plt.bar(range(len(tptnfpfn)), values2, tick_label=names2)
plt.show()'''
"""testimonial = TextBlob(text)
#Here text means the sentence without emojis
if ((testimonial.sentiment.polarity > 0 and emojipol < 0) or(testimonial.sentiment.polarity < 0 and emojipol > 0) or (testimonial.sentiment.polarity == 0 and emojipol < 0) or (testimonial.sentiment.polarity == 0 and emojipol > 0)):#sentence is sarcast"""