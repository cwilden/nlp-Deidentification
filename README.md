# nlp-Deidentification
De-identification of PHI in Medical Records

We present a baseline system and improvements in order to obtain better results for de-identification of PHI in medical records provided by the 2014 I2B2 (Center of Informatics for Integrating Biology and Bedside) challenge. 

Dataset is now available for download on Harvard Medical school’s DBMI Data Portal, under n2c2 NLP Research Data Sets. The dataset for our research is the 2014 De-identification and Heart Disease Risk Factors Challenge. The input files are XML files. The way they are structured is they have a header with some information followed by a free-text discharge summary. 

The output of the dataset should be the PHI main categories such as: Name (Patient, Doctor, Username), Profession, Location (Country, Organization, ZIP Code), Age, Date, Contact (Phone, Fax, Email, Ipaddr), and ID (MedicalRecord).

The baseline system is a combination of Conditional Random Fields (CRF) and a rule-based classifier. The preprocessing required is first extracting the raw clinical notes in the input XML files. Then sentence splitting followed by tokenization. There is some preprocessing that occur for specific features. The total features used for our system are:

•	Bag-of-words (unigram, bigram, and trigram)

•	Part-of-speech (POS) tags (NLTK) 

•	Combination of tokens and POS tags [w0 p-1, w0 p0, w0 p1, w0 p-1 p0, w0 p0 p1, w0 p-1 p1, w0 p-1 p0 p1] where w0 is the current word and p-1 p0 p1 are previous, current, and next POS tags.

•	Sentence information (length of sentence, punctuation present)

•	Affixes (prefix and suffix lengths: 1 – 5)

•	Word shapes (A is upper, a is lower, # is digit, - is punctuation)

•	Section information (what section is this text under? Ex. “Medications”)

•	Stanford Named Entity Recognition tags

•	Word2Vec with lemmatized words

•	Dictionary features (Country, State, City, ZIP code)

