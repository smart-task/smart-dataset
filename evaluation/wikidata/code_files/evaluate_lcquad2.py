import json
import pandas as pd
import codecs
import datetime

#class for evaluation
class fc_eval:
     def __init__(self):
        #count for dumy questions having predicted answers
        self.cntr_dummy_ques = 0

     #function for evaluation if category is boolean
     def fetch_rank_bool(self, catg_pred, catg_correct):
         if catg_pred == catg_correct:
            rank = 1
         else:
            rank = 0
         return rank

     #function for evaluation if category is literal or resource
     def fetch_rank_literal_resource(self, catg_pred, catg_correct, lst_pred, lst_correct):
         if catg_pred == catg_correct:
            #print('correct list is:')
            #print(lst_correct)
            #print('pred is')
            #print(lst_pred)
            for j in range(len(lst_pred)):           
                if (lst_pred[j]) in str(lst_correct):
                   #print(lst_pred[j] + ' is present in correct list')
                   rank = lst_correct.index(lst_pred[j]) + 1
                   break
                rank = 0            
         else:
            rank = 0
         return rank

     

     #function to evaluate answer types
     def evaluate(self, inp_file_correct, inp_file_pred, out_file_path, results_file):
        json_collection = []
        #opening dummy predictions file
        with codecs.open(inp_file_pred, 'r', 'utf-8-sig') as data_file: 
            json_data = json.load(data_file)
            pred_data_df = pd.DataFrame(json_data)
        #opening correct lcquad2 ans type file
        with codecs.open(inp_file_correct, 'r', 'utf-8-sig') as df_correct: 
             j_data_correct = json.load(df_correct)
             correct_data_df = pd.DataFrame(j_data_correct)
        cntr_ques_matched = 0
        catg_matched = 0
        rank = 0
        rr = 0
        sum_rr = 0
        mrr = 0
        #iterating through the dummy predictions file
        for i in range(len(pred_data_df['id'])):
            self.cntr_dummy_ques += 1
            #creating a reduced data frame out of entire correct data file, which had data of only 1 concerned question in each iteration
            sel_df = correct_data_df.loc[correct_data_df['id']==pred_data_df['id'][i]]
            if not(sel_df.empty):
               #checking if questions are same in dummy prediction file and correct file
               if (pred_data_df['id'][i] == list(sel_df['id'])[0]):
                  cntr_ques_matched += 1
                  #print('question matched')
                  #if category does not match, then assigning rank = 0
                  if (pred_data_df['category'][i] != list(sel_df['category'])[0]):
                     rank = 0
                  #if category matches
                  else:
                     catg_matched += 1
                     #print('category matched')
                     if list(sel_df['category'])[0] == 'boolean':
                        rank = self.fetch_rank_bool(pred_data_df['category'][i], list(sel_df['category'])[0])
                     elif (list(sel_df['category'])[0] == 'literal' or list(sel_df['category'])[0] == 'resource'):
                        #print('not boolean')
                        rank = self.fetch_rank_literal_resource(pred_data_df['category'][i], list(sel_df['category'])[0], list(pred_data_df['type'][i]), list(sel_df['type'])[0])
                    
            #in case correct ans typ file has no question matching the file with dummy question predictions
            else:
              rank = 0                
            #assigning values to reciprocal rank i.e. to rr
            if rank!= 0:
               rr = 1/rank
            else:
               rr = 0

            #framing output file format
            try:      
                     data = {}
                     try:
                         data['id'] = int(pred_data_df['id'][i])
                     except:
                         data['id'] = ''
                         pass
                     try:
                         data['question'] = pred_data_df['question'][i]
                     except:
                         data['question'] = ''
                         pass
                     try:
                         data['category'] = pred_data_df['category'][i]
                     except:
                         data['category'] = ''
                         pass
                     try:
                         data['type'] = pred_data_df['type'][i]
                     except:
                         data['type'] = ''
                         pass
                     try:
                         data['rank'] = rank
                     except:
                         data['rank'] = ''
                         pass
                     try:
                         data['rr'] = rr
                     except:
                         data['rr'] = ''
                         pass
                     json_collection.append(data)
            except Exception as ex:
                   print(ex)
                   pass
            #summing reciprocal rank for calculation of Mean Reciprocal Rank(mrr)
            sum_rr += rr
        #print(str(self.cntr_dummy_ques)+' total predicted question types exist in original input file')
        #print(str(cntr_ques_matched)+' correct question matches exist')
        #print('No correct question matches exist for '+str(self.cntr_dummy_ques - cntr_ques_matched)+' questions')
        catg_accuracy = round((catg_matched/self.cntr_dummy_ques),2)
        mrr = round((sum_rr/cntr_ques_matched),2)
        print('catg_matched = '+str(catg_accuracy))
        print('mrr = '+str(mrr))

        #closing input data files
        data_file.close()
        df_correct.close()

        #writing out the json file appended with rank and rr
        jf = open(out_file_path,'w')
        json.dump(json_collection,jf,indent=6)
        jf.close()
 
        #writing out the results file having category accuracy and mrr
        json_res_collection = []
        dat = {}
        dat['category_matched_accuracy'] = catg_accuracy
        dat['type_mrr'] = mrr
        json_res_collection.append(dat)
          
        results_path = results_file + str(datetime.datetime.now()) + '.json'
        jf = open(results_path,'w')
        json.dump(json_res_collection,jf,indent=2)
        jf.close()

if __name__ == '__main__':
   obj = fc_eval()
   inp_file_path1 = '../data_files/lcquad2_gold_standard.json'
   inp_file_path2 = '../data_files/lcquad2_full_predictions.json'                            #lcquad2_dummy_predictions.json
   out_file_path = '../data_files/lcquad2_full_predictions_with_rr.json'                     #lcquad2_dummy_predictions_with_rr.json
   results_file = '../data_files/final_scores_'
   obj.evaluate(inp_file_path1,inp_file_path2,out_file_path,results_file)

