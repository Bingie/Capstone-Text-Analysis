
import pandas as pd
import numpy as np
import pprint as pp

def drop_columns(df, idx = []):
    
    df.drop(df.columns[idx], axis = 1, inplace = True)
    
    return df
    

def select_comments(df, keywords = ["comment"]):
     
    dict_col_comments = {}
    for i, col in enumerate(df.columns):
         for word in keywords:
             if word.lower() in col.lower():
                 dict_col_comments[col] = i  
                               
    
    comments = df[dict_col_comments.keys()].dropna(how = "all")
    
    df = df.loc[comments.index]
    
    return df, dict_col_comments
    

def cleaning_comments(df, min_words = 1, column = "Comment"):
    
    
    df[column] = df[column].apply(lambda y: np.nan if len(y.split()) <= min_words else y)
    idx = df[column].dropna().index
    df = df.loc[idx,:]
    
    return df


def rename_columns(df, rename_dictionary):
    
    column_names_dict = {} 
    for k,v in rename_dictionary.items():
        column_names_dict[df.columns[k]] = v
    
    df = df.rename(column_names_dict, axis = 1)
    
    return df
    
def data_cleaning(df):
    
    
    """ Data cleaning - Dataset 4
    
    df = df.dropna(how = "all",axis = 1)
    df.loc[:,"Employee_Index"] = df.index.values 
    
    df, position_comments = select_comments(df, keywords = ["comments"])
    
    position = df.iloc[:,8].str.split("(")
    df.iloc[:,8] = position.apply(lambda x: x[0].strip())
    
    drop_columns_idx = np.r_[0:2]
    df = drop_columns(df, drop_columns_idx)
    
    rename_dict = {-2: "Comment"}
    df = rename_columns(df, rename_dict)
    
    df = df.loc[df["Comment"].dropna().index] 
    df = cleaning_comments(df, column = "Comment")
    df.to_excel("./raw data/data-company4-cleaned.xlsx")
    
    
    return df
    """
     
    """
    #Data Cleaning - Dataset 3
    df = df.dropna(how = "all",axis = 1)
    
    Titles = df.iloc[:,2:6].fillna('').astype(str)
    df.loc[:,("Title")] = Titles.values.sum(axis = 1)
    df.loc[:,"Employee_Index"] = df.index.values 
    
    drop_columns_idx = np.r_[1:6, 8:9]
    df = drop_columns(df, drop_columns_idx)
    
    df, position_comments = select_comments(df, keywords = ["comment","why","additional"])
    
    rename_dict = {0: "Area", 1: "Gender", 48: "Workplace Flexibility", 
                   49: "General Comments", 50: "Career Advancement",
                   51: "Women's experience", 11: "Gender Diversity",
                   19: "Gender Balance", 28: "Coaching Programs", 43: "Commentaries"}
    
    df = rename_columns(df, rename_dict)
    
    unpivot_columns = {"Workplace Flexibility", "General Comments", 
                       "Career Advancement", "Women's experience", "Gender Diversity",
                       "Gender Balance", "Coaching Programs", "Commentaries"}
    
    preserve_columns = [col for col in df.columns if col not in unpivot_columns]
    
    df = pd.melt(df, id_vars = preserve_columns, 
                 value_vars = unpivot_columns, 
                 var_name="Dimension", value_name="Comment")
    
    df = df.loc[df["Comment"].dropna().index] 
    df = cleaning_comments(df)
    df.to_excel("./raw data/data-company3-cleaned.xlsx")
    
    return df
    """
    
    #Data Cleaning dataset 2 
    
    df = df.dropna(how = "all",axis = 1)
    
    drop_columns_idx = np.r_[0:6, 10]
    df = drop_columns(df, drop_columns_idx)
    
    area = df.iloc[:,0].str.split("(")
    df.iloc[:,0] = area.apply(lambda x: x[0].strip())
    
    df.loc[:,"Supervisor"] = area.apply(lambda x: x[1][:x[1].index(")")])
    
    position = df.iloc[:,4].str.split("(")
    df.iloc[:,4] = position.apply(lambda x: x[0].strip())
    
    df, position_comments = select_comments(df, keywords = ["comment"])
        
    rename_dict = {0: "Area", 1: "Location", 2: "Duration", 3: "Gender",
                   4: "Position", -2: "Suggestions", -3: "Workplace Flexibility",
                   -4: "Career Advancement", -5: "Career Development", 
                   -6: "Workplace Environment", -7: "Gender Diversity"}
    
    df = rename_columns(df, rename_dict)
    df.loc[:,"Employee_Index"] = df.index.values 
    
    pivot_columns = ["Suggestions", "Workplace Flexibility",
                     "Career Advancement", "Career Development", 
                     "Workplace Environment", "Gender Diversity"]
    
    preserve_columns = [col for col in df.columns if col not in pivot_columns]
    
    df = pd.melt(df, id_vars = preserve_columns, 
                 value_vars = pivot_columns, 
                 var_name="Dimension", value_name="Comment")
    
    df = df.loc[df["Comment"].dropna().index] 
    df = cleaning_comments(df)
    df.to_excel("./raw data/data-company2-cleaned.xlsx")
    
    return df
    


if __name__ == "__main__":
    filepath = "J:/Users/John/Documents/ITC Capstone/FastText/raw data/Data-Company 2.csv"
    df = pd.read_csv(filepath, delimiter = ",")
    clean_df = data_cleaning(df)    





    
    
    
    
    