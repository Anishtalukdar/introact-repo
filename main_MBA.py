from flask import Flask, request, redirect, url_for, flash, jsonify
from flask_restful import Resource
import pandas as pd
import os
import json
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#added for blueprint
from flask import Blueprint

#added for blueprint
mbaautoml_api = Blueprint('mbaautoml_api', __name__)

#app = Flask(__name__)

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
@mbaautoml_api.route('/api/mb_Analysis', methods=['POST'])
def MBAnalysis():
        file_path = request.json['file_path']
        #min_support = request.json['min_support']
        metric = request.json['metric']
        min_threshold=request.json['min_threshold']
        
        df = pd.read_csv(file_path)
        print(df.head())
        
        #sns.countplot(x = 'Description', data = df, order = df['Description'].value_counts().iloc[:10].index)
        try:
            df['Description'] = df['Description'].str.strip()
            df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
            df['InvoiceNo'] = df['InvoiceNo'].astype('str')
            df = df[~df['InvoiceNo'].str.contains('C')]
            
            basket = (df[df['Country'] =="France"]
                      .groupby(['InvoiceNo', 'Description'])['Quantity']
                      .sum().unstack().reset_index().fillna(0)
                      .set_index('InvoiceNo'))
            
            #te = TransactionEncoder()
            #te_ary = te.fit(df).transform(df)
            #basket_sets = pd.DataFrame(te_ary, columns=te.columns_)
            
            def encode_units(x):
                if x <= 0:
                    return 0
                if x >= 1:
                    return 1
            
            basket_sets = basket.applymap(encode_units)
            basket_sets.drop('POSTAGE', inplace=True, axis=1)
            
            #basket_sets.to_csv("E:\\Product_transaction.csv")
            
            frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
            
            rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
            
            rules.drop(columns=['antecedent support','consequent support','leverage','conviction'], inplace=True)
            rules.rename(columns={'antecedents':'Key Products','consequents':'Associated Products'}, inplace = True)
            rules_toJson = rules.to_json(orient='records')
            
            #rules.to_csv("Rules.csv", index=False)
            #rules.to_json("Rules.json")
            rules.head()
            print(rules_toJson)
            return rules_toJson
        except Exception as e:          
            error_statement = str(e)
            print("Error statement: ",error_statement)
            return error_statement
        
if __name__ == '__main__':
    #app.run(host='localhost', debug=True, port=5000)
    app.run(host='10.12.1.206', debug=True, port=54321)

