from pipelines import pipeline 
import pickle 
nlp = pipeline("question-generation")
pickle.dump(nlp, open('model.pkl','wb'))
model = pickle.load(open('model.pkl', 'rb'))
text = "Apple was founded in the year 1970. Its Headquarters are located in California."
#print(model(text))
from transformers import pipeline
summarizer = pipeline("summarization")
text2 = """
Saving and loading a general checkpoint model for inference or resuming training can be helpful for picking up where you last left off. When saving a general checkpoint, you must save more than just the model’s state_dict. It is important to also save the optimizer’s state_dict, as this contains buffers and parameters that are updated as the model
 trains. Other items that you may want to save are the epoch you left
  off on, the latest recorded training loss, external torch.nn.Embedding layers, and more, based on your own algorithm.
"""
pickle.dump(summarizer, open('model_summ.pkl', 'wb'))
#print(summarizer(text2))
model_summ = pickle.load(open('model_summ.pkl', 'rb'))
# print("Summary: ", model_summ("The sky is blue."))
