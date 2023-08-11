from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pickle
import PyPDF2
import nltk
import re
nltk.download('punkt')

import spacy
import PyPDF2.errors


# Load the pre-trained model and vectorizer
model = pickle.load(open("Resume_Classifier1.pkl", "rb"))
vect = pickle.load(open("trained_vectorizer.pkl", "rb"))

# Load the Spacy model for natural language processing
nlp = spacy.load('en_core_web_sm')

# Define the Flask app
app = Flask(__name__, static_url_path='/static')

# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the uploaded file from the HTML form
        file = request.files['file']
        try:
            # Load the PDF file
            pdf_reader = PyPDF2.PdfReader(file)
            resume_text = ''
            for page in pdf_reader.pages:
                resume_text += page.extract_text()

            # Clean the text
            resume_text = resume_text.replace('\n', ' ')
            resume_text = resume_text.replace('\t', ' ')

            # Extract the objective summary
            sentences = nltk.sent_tokenize(resume_text)
            pattern = re.compile(r'(objective|summary)?\s*(.*)', re.IGNORECASE)

            objective_summary = None
            for sentence in sentences:
                match = pattern.match(sentence)
                if match:
                    objective_summary = match.group(2)
                    break

              # Check if objective_summary is None
            if objective_summary is None:
                objective_summary = ''
            
            # Extract the key skills from the resume text
            skills = ['Python','pandas', 'numpy', 'scipy', 'scikit-learn','matplotlib','Sql', 'Java', 'JavaScript/JQuery','Machine learning','Regression', 'SVM', 'Naive Bayes', 'KNN', 'Random Forest', 'Decision Trees', 'Boosting techniques', 'Cluster Analysis', 'Word Embedding', 'Sentiment Analysis', 'Natural Language processing', 'Dimensionality reduction', 'Topic Modelling' '(LDA, NMF)','PCA & Neural Nets','Database Visualizations','Mysql','SqlServer','Cassandra','Hbase', 'ElasticSearch','D3.js', 'DC.js', 'Plotly','kibana','matplotlib', 'ggplot', 'Tableau','Regular Expression', 'HTML', 'CSS', 'Angular', 'Logstash', 'Kafka', 'Python Flask', 'Git, Docker', 'computer vision',"Windows XP","Ms Office","Word","Excel","Look-ups","Pivot table","Power Point",'Auto CAD','CNC Programming','Photoshop','MS office','PowerPoint','Catia','Autocad','Photoshop INDUSTRIAL EXPOSURE','Linux','Derby (Embedded DB)','Eclipse','SonarQube','Putty','Bizagi', 'MS Visio Prototyping Tool','Indigo Studio','Core Java','Data Warehousing','SAGE', 'Flotilla', 'LM' 'ERP', 'Tally 9', 'WMS', 'Exceed 4000','Scrum Version Control','Sqlite','MongoDB','PHP','JavaScript','Ajax','Jquery','XML', 'Agile Methodology', 'DevOps Methodology','Scrum Framework', 'JIRA Tool', 'GIT', 'Bitbucket','Windows', 'Linux', 'Ubuntu Network Technologies','Cisco Routing and Switching', 'InterVLAN Routing', 'Dynamic Protocols','RIPv2', 'RIPng', 'OSPF', 'EIGRP','Static Routing', 'ACL', 'VTP', 'VLAN', 'EhterChannel', 'HSRP', 'STP', 'IPv6','postgreSQL Windows', 'Linux Putty','Hadoop', 'Map Reduce', 'HDFS', 'Hive', 'Sqoop','Talend Big Data','Microsoft SQL Server','SQL Pla Studio Workbench','AWS Services','CSS Testing Manual Testing', 'Database Testing Other Bug tracking']
            key_skills = []
            for token in nlp(resume_text):
                if token.text.lower() in [s.lower() for s in skills]:
                    if token.text.lower() not in [ks.lower() for ks in key_skills]:
                        key_skills.append(token.text)
            key_skills_str = ' '.join(key_skills)

            # Combine the objective summary and key skills into a single string
            final = objective_summary + ' ' + key_skills_str

            # Clean the text using regular expressions
            final = re.sub(r'[^\w\s]|\.','', final)
            final = re.sub(r'\s+', ' ', final)
            final = final.lower()
            final = final.strip()
            print (final)

            # Transform the single string into a sparse matrix of TF-IDF features
            single_string_features = vect.transform([final])

            # Classify the resume into one of the job categories
            classes = {
                0:'Advocate',
                1:'Arts',
                2: 'Automation Testing',
                3:'Blockchain ',
                4:'Business Analyst',
                5:'Civil Engineer',
                6:'Data Science',
                7:'Database',
                8:'DevOps Engineer',
                9:'DotNet Developer',
                10:'ETL Developer',
                11:'Electrical Engineering',
                12:'HR',
                13:'Hadoop',
                14:'Health and fitness',
                15:'Java Developer',
                16:'Mechanical Engineer',
                17:'Network Security Engineer',
                18:'Operations Manager',
                19:'PMO',
                20:'Python developer',
                21:'SAP developer',
                22:'Sales',
                23:'Web Designing'
            }
            prediction = model.predict(single_string_features)
            result = classes[prediction[0]]

            # Render the result page
            return render_template('results.html', result=result)
        except PyPDF2.errors.EmptyFileError:
            #Handle empty file error
            error_msg = "Error: Empty file uploaded. Please choose a file to upload."
            return render_template('index.html', error_msg=error_msg)

    else:
        return render_template('index.html')
if __name__== '__main__':
    app.run(host='0.0.0.0',port=8080,debug=True)       
        