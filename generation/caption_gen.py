#imports
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import base64
import anthropic
import requests


class gen_cap:
    
    def __init__(self, gpt_key = None, claude_key = None, data_h = None, data_g = None):
        
        self.gpt_key = gpt_key
        self.claude_key = claude_key
        self.data_h = pd.read_csv(data_h) #the full hessle dataset
        self.data_g = pd.read_csv(data_g) #the full gpt generate dataset
            
    def gpt_35_turbo(self, contest_number = []):
        '''
        using model gpt 3.5 turbo to generate captions for each cartoon.
        input: a list of contest number that should be in the hessle dataset.
        output: a datafram containg each contest number and their respected 10 captions. 
        '''
        index = []
        for i in contest_number:
            index.append(self.data_h['contest_number'].tolist().index(i))
        #generate caption
        result = {}
        client = OpenAI(api_key = self.gpt_key)
        for i in index:
            n = self.data_h['contest_number'].iloc[i]
            c = self.data_h['image_description'].iloc[i]
            u = self.data_h['image_uncanny_description'].iloc[i]
            l = self.data_h['image_location'].iloc[i]
            e = ""
            for i in self.data_h['entities'].iloc[0]:
                e += i.split('/')[-1] + ', '
            e = e[:-2] 
            
            response = client.chat.completions.create(
              model="gpt-3.5-turbo",
              messages = [
            {"role": "system", "content": "I want you to act as a sophisticated reader of The New Yorker Magazine. You are competing in The New Yorker Cartoon Caption Contest.  Your task is to generate funny captions for a cartoon. Here are some ideas for developing funny captions.   First think about characteristics associated with the objects and people featured in the cartoon. Then consider what are the unusual or absurd elements in the cartoon. It might help to imagine conversations between the characters.  Then think about funny and non-obvious connections that can be made between the objects and characters. Try to come up with funny captions that fit the cartoon, but are not too direct.  It may be funnier if the person reading the caption has to think a little bit to get the joke. Next, I will describe a cartoon image and then you should generate 10 funny captions for the cartoon along with an explanation for each."},
            {
              "role": "user",
              "content": [
                {"type": "text", "text": "The cartoon's description is: " + c + ".\n The uncanny description is: " + u + ".\n The location of the cartoon is:" + l + ".\n The entities of the cartoon are: " + e + "."},
              ],
            }
              ],
            max_tokens=1000,
            )
            try:
                result[n] = response.choices[0].message.content
            except:
                result[n] = "e"
                
            #the processing need to be manual check and reruned
            processed = {}
            for i in result:
                curr = []
                for j in result[i].split("\n"):
                    try:
                        int(j[0])
                        pend = j[4:-1]
                        pend = pend.replace("*", "")
                        pend = pend.replace("Caption", "")
                        pend = pend.replace(":", "")
                        pend = pend.replace("aption \"", "")
                        curr.append(pend)
                    except:
                        continue
                processed[i] = curr
            return processed    
        
    def gpt_4o(self, contest_number = []):
        '''
        using model gpt 4o to generate captions for each cartoon.
        input: a list of contest number that should be in the hessle dataset.
        output: a datafram containg each contest number and their respected 10 captions. 
        '''
        index = []
        for i in contest_number:
            index.append(self.data_h['contest_number'].tolist().index(i))
        #generate caption
        result = {}
        client = OpenAI(api_key = self.gpt_key)
        for i in index:
            n = self.data_h['contest_number'].iloc[i]
            c = self.data_h['image_description'].iloc[i]
            u = self.data_h['image_uncanny_description'].iloc[i]
            l = self.data_h['image_location'].iloc[i]
            e = ""
            for i in self.data_h['entities'].iloc[0]:
                e += i.split('/')[-1] + ', '
            e = e[:-2] 
            
            response = client.chat.completions.create(
              model="gpt-4o",
              messages = [
            {"role": "system", "content": "I want you to act as a sophisticated reader of The New Yorker Magazine. You are competing in The New Yorker Cartoon Caption Contest.  Your task is to generate funny captions for a cartoon. Here are some ideas for developing funny captions.   First think about characteristics associated with the objects and people featured in the cartoon. Then consider what are the unusual or absurd elements in the cartoon. It might help to imagine conversations between the characters.  Then think about funny and non-obvious connections that can be made between the objects and characters. Try to come up with funny captions that fit the cartoon, but are not too direct.  It may be funnier if the person reading the caption has to think a little bit to get the joke. Next, I will describe a cartoon image and then you should generate 10 funny captions for the cartoon along with an explanation for each."},
            {
              "role": "user",
              "content": [
                {"type": "text", "text": "The cartoon's description is: " + c + ".\n The uncanny description is: " + u + ".\n The location of the cartoon is:" + l + ".\n The entities of the cartoon are: " + e + "."},
              ],
            }
              ],
            max_tokens=1000,
            )
            try:
                result[n] = response.choices[0].message.content
            except:
                result[n] = "e"
                
            #the processing need to be manual check and reruned
            processed = {}
            for i in result:
                curr = []
                for j in result[i].split("\n"):
                    try:
                        int(j[0])
                        pend = j[4:-1]
                        pend = pend.replace("*", "")
                        pend = pend.replace("Caption", "")
                        pend = pend.replace(":", "")
                        pend = pend.replace("aption \"", "")
                        curr.append(pend)
                    except:
                        continue
                processed[i] = curr
            return processed    

    def gpt_4o_vision(self, contest_number = []):
        '''
        using model gpt 4o to generate captions for each cartoon.
        input: a list of contest number that should be in the hessle dataset. 
        output: a datafram containg each contest number and their respected 10 captions. 
        '''
        index = []
        for i in contest_number:
            index.append(self.data_h['contest_number'].tolist().index(i))
        #generate caption
        result = {}
        client = OpenAI(api_key = self.gpt_key)
        for i in self.index:
            n = self.data_h['contest_number'].iloc[i]
            c = self.data_h['image_description'].iloc[i]
            u = self.data_h['image_uncanny_description'].iloc[i]
            l = self.data_h['image_location'].iloc[i]
            e = ""
            for i in self.data_h['entities'].iloc[0]:
                e += i.split('/')[-1] + ', '
            e = e[:-2] 
            
            response = client.chat.completions.create(
              model="gpt-4o",
              messages = [
            {"role": "system", "content": "I want you to act as a sophisticated reader of The New Yorker Magazine. You are competing in The New Yorker Cartoon Caption Contest.  Your task is to generate funny captions for a cartoon. Here are some ideas for developing funny captions.   First think about characteristics associated with the objects and people featured in the cartoon. Then consider what are the unusual or absurd elements in the cartoon. It might help to imagine conversations between the characters.  Then think about funny and non-obvious connections that can be made between the objects and characters. Try to come up with funny captions that fit the cartoon, but are not too direct.  It may be funnier if the person reading the caption has to think a little bit to get the joke. Next, I will describe a cartoon image and then you should generate 10 funny captions for the cartoon along with an explanation for each."},
            {
              "role": "user",
              "content": [
                {"type": "text", "text": "The cartoon's description is: " + c + ".\n The uncanny description is: " + u + ".\n The location of the cartoon is:" + l + ".\n The entities of the cartoon are: " + e + ".\n Here is the cartoon itself."},
                {
                  "type": "image_url",
                  "image_url": {
                    "url": "https://nextml.github.io/caption-contest-data/cartoons/"+ str(n) +".jpg",  "detail": "high",
                  },
                },
              ],
            }
              ],
            max_tokens=1000,
            )
            try:
                result[n] = response.choices[0].message.content
            except:
                result[n] = "e"
                
            #the processing need to be manual check and reruned
            processed = {}
            for i in result:
                curr = []
                for j in result[i].split("\n"):
                    try:
                        int(j[0])
                        pend = j[4:-1]
                        pend = pend.replace("*", "")
                        pend = pend.replace("Caption", "")
                        pend = pend.replace(":", "")
                        pend = pend.replace("aption \"", "")
                        curr.append(pend)
                    except:
                        continue
                processed[i] = curr
            return processed    
        
    def claude_vision(self, contest_number = []):
        '''
        using model claude 3 opus to generate captions for each cartoon.
        input: a list of contest number that should be in the hessle dataset. 
        output: a datafram containg each contest number and their respected 10 captions. 
        '''
        index = []
        for i in contest_number:
            index.append(self.data_h['contest_number'].tolist().index(i))
            
        result = {}
        client = anthropic.Anthropic(
        api_key= self.claude_key,
        )
        for i in self.index:
            n = self.data_h['contest_number'].iloc[i]
            c = self.data_h['image_description'].iloc[i]
            u = self.data_h['image_uncanny_description'].iloc[i]
            l = self.data_h['image_location'].iloc[i]
            e = ""
            for i in self.data_h['entities'].iloc[0]:
                e += i.split('/')[-1] + ', '
            e = e[:-2] 
            img_b64 = base64.b64encode(requests.get("https://nextml.github.io/caption-contest-data/cartoons/"+ str(n)+".jpg").content).decode("ascii")


            message = client.messages.create(
            model="claude-3-opus-20240229",
            system =  "I want you to act as a sophisticated reader of The New Yorker Magazine. You are competing in The New Yorker Cartoon Caption Contest.  Your task is to generate funny captions for a cartoon. Here are some ideas for developing funny captions.   First think about characteristics associated with the objects and people featured in the cartoon. Then consider what are the unusual or absurd elements in the cartoon. It might help to imagine conversations between the characters.  Then think about funny and non-obvious connections that can be made between the objects and characters. Try to come up with funny captions that fit the cartoon, but are not too direct.  It may be funnier if the person reading the caption has to think a little bit to get the joke. Next, I will describe a cartoon image and then you should generate 10 funny captions for the cartoon along with an explanation for each.",     
            max_tokens=1000,
            messages=[
                    {
            "role": "user",
            "content": [
            {"type": "text", "text": "The cartoon's description is: " + c + ".\n The uncanny description is: " + u + ".\n The location of the cartoon is:" + l + ".\n The entities of the cartoon are: " + e + ".\n Here is the cartoon itself."},
            {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        },
                    },
            ],
            }
                 ])

            try:
                result[n] = message.content[0].text
            except:
                result[n] = "e"
            
            #the processing need to be manual check and reruned
            processed = {}
            for i in result:
                curr = []
                for j in result[i].split("\n"):
                    try:
                        int(j[0])
                        pend = j[4:-1]
                        pend = pend.replace("*", "")
                        pend = pend.replace("Caption", "")
                        pend = pend.replace(":", "")
                        pend = pend.replace("aption \"", "")
                        curr.append(pend)
                    except:
                        continue
                processed[i] = curr
            return processed    
