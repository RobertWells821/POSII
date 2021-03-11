from sklearn.metrics import plot_confusion_matrix
import json
import numpy as np												#Математическая библиотека

#Методы машинного обучения
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import LinearSVC as SVC 
from sklearn.naive_bayes import MultinomialNB as NB
from sklearn.tree import DecisionTreeClassifier as C4_5

from sklearn.metrics import roc_curve as roc, auc, confusion_matrix, classification_report, precision_recall_curve, ConfusionMatrixDisplay			#ROC-AUC кривые
import matplotlib.pyplot as plt								#Библиотека графических выводов
from sklearn.model_selection import train_test_split		#Класс для разделения базы данных

def rr(file_):
    name1=['udp', 'tcp', 'icmp']
    name2=['time', 'netbios_ssn', 'urh_i', 'pop_3', 'tftp_u', 'discard', 'vmnet', 'ecr_i', 'klogin', 'gopher', 'uucp', 'whois', 'urp_i', 'other', 'IRC', 'exec', 'login', 'printer', 'systat', 'domain_u', 'ntp_u', 'kshell', 'http_443', 'ssh', 'uucp_path', 'sql_net', 'imap4', 'X11', 'sunrpc', 'eco_i', 'ctf', 'daytime', 'nntp', 'auth', 'smtp', 'pop_2', 'netstat', 'mtp', 'private', 'red_i', 'hostnames', 'finger', 'netbios_ns', 'domain', 'nnsp', 'telnet', 'pm_dump', 'http', 'csnet_ns', 'rje', 'name', 'netbios_dgm', 'ftp', 'iso_tsap', 'Z39_50', 'tim_i', 'courier', 'ftp_data', 'shell', 'supdup', 'echo', 'bgp', 'link', 'ldap', 'remote_job', 'efs']
    name3=['OTH', 'S1', 'RSTOS0', 'SF', 'S0', 'RSTR', 'RSTO', 'S2', 'SH', 'REJ', 'S3']
    target_names=['back', 'buffer_overflow', 'ftp_write', 'guess_passwd', 'imap', 'ipsweep', 'land', 'loadmodule', 'multihop', 'neptune', 'nmap', 'normal', 'perl', 'phf', 'pod', 'portsweep', 'rootkit', 'satan', 'smurf', 'spy', 'teardrop', 'warezclient', 'warezmaster']
    p=open(file_).read().split('\n')
    x, y = [], []
    for i in p:
        if len(i)>1:
            if i[-1]=='.':d=i[:-1].split(',')
            else: d=i.split(',')
        if len(d)>40 and len(d)<45:
            dd=[]
            for j in range(len(d)-1):
                if j==1: dd.append(name1.index(d[j]))
                elif j==2: dd.append(name2.index(d[j]))
                elif j==3: dd.append(name3.index(d[j]))
                else: dd.append(d[j])
            x.append(dd)
            y.append(target_names.index(d[-1]))
    return np.array(x).astype(np.float64), np.array(y).astype(np.float64)

def rn(p, y_test):
	if len(y_test)==len(p):
		t=0
		for i in range(len(y_test)):
			if y_test[i]!=p[i]:t+=1
		print("Correct: {} \nNot correct: {}\nPercent: {}".format(len(y_test)-t,t, 100-t/len(y_test)*100))
		print(classification_report(y_test, p))
		# print(confusion_matrix(y_test, p))
 

def main():
	X, y = rr('kddcup.data_10_percent')
	X, X_test, y, y_test = train_test_split(X, y, random_state = 75)
	y=np.array([0 if i != 11 else 1 for i in y])
	yy=np.array([0 if i != 11 else 1 for i in y_test])
	random_state = np.random.RandomState(0)
	kdd3 =  SVC()
	kdd2 =  C4_5(random_state=0)
	kdd1 = NB()
	kdd =  RF(n_estimators=100, random_state=0, max_features=2)
	p3 = kdd3.fit(X,y)
	p2 = kdd2.fit(X,y)
	p1 = kdd1.fit(X,y)
	p = kdd.fit(X,y)
	t=p.predict(X_test)
	t1=p1.predict(X_test)
	t2=p2.predict(X_test)
	t3=p3.predict(X_test)
	print("Random Forest")
	rn(t,yy)
	print("Native Base")
	rn(t1,yy)
	print("C4.5")
	rn(t2,yy) 
	print("SVC")
	rn(t3,yy)

	# Построение матрицы ошибок 
	cm3=confusion_matrix(yy, t3)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm3)
	disp.plot()
	plt.show()
	cm2=confusion_matrix(yy, t2)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm2)
	disp.plot()
	plt.show()
	cm1=confusion_matrix(yy, t1)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm1)
	disp.plot()
	plt.show()
	cm=confusion_matrix(yy, t)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm)
	disp.plot()
	plt.show()


	# Построение AUC-ROC кривых 
	pr3, rc3, _ = precision_recall_curve(yy, t3)
	fpr, tpr, _ = roc(yy, p3.decision_function(X_test))
	roc_auc  = auc(fpr, tpr)
	plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('SVM', roc_auc))

	pr2, rc2, _ = precision_recall_curve(yy, t2)
	fpr, tpr, _ = roc(yy, p2.predict_proba(X_test)[:, 1])
	roc_auc  = auc(fpr, tpr)
	plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('C4.5', roc_auc))

	pr1, rc1, _ = precision_recall_curve(yy, t1)
	fpr, tpr, _ = roc(yy, p1.predict_proba(X_test)[:, 1])
	roc_auc  = auc(fpr, tpr)
	plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('NB', roc_auc))

	pr, rc, _ = precision_recall_curve(yy, t)
	fpr, tpr, _ = roc(yy, p.predict_proba(X_test)[:, 1])
	roc_auc  = auc(fpr, tpr)
	plt.plot(fpr, tpr, linestyle='--', label='%s ROC (area = %0.2f)' % ('RF', roc_auc))


	plt.plot([0, 1], [0, 1], linestyle='--')
	plt.legend(loc=0, fontsize='small')
	plt.show()
	plt.plot(rc3, pr3, label='SVM')
	plt.plot(rc2, pr2, label='C4.5')
	plt.plot(rc1, pr1, label='NB')
	plt.plot(rc, pr, label='RF')
	plt.legend(loc=0, fontsize='small')
	plt.show()


if __name__=='__main__':
		main()
