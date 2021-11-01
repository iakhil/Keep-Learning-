from pipelines import pipeline 
import pickle
ans = pipeline("multitask-qa-qg") 
pickle.dump(ans, open('answer_model.pkl','wb'))
model = pickle.load(open('answer_model.pkl', 'rb'))
text = "Apple was founded in the year 1970. Its Headquarters are located in California."
#print(model(text))
text2 = """
        The sky is blue.
"""
# print("Summary: ", model_summ("The sky is blue."))
qa = ans({
    'question': "What colour is the sky?", 
    'context': text2
})

print(qa) 
