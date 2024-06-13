import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import base64
import ast
import requests


class gen_des:
    
    def __init__(self, gpt_key = None, data_h = None, data_g = None):
        
        self.gpt_key = gpt_key
        self.data_h = pd.read_csv(data_h) #the full hessle dataset
        self.data_g = pd.read_csv(data_g) #the full gpt generate dataset
        self.shots = self.data_h.sample(n=5, random_state = 36) #the shots for 5 shots
        
    def description_gpt4_o(self, contest_number = []):
        '''
        use gpt-4o model to gerenate cartoon description, using 5 shot methods
        input: a list of contest number
        output: a datafram contaning the contest number and their respected canny and uncanny description, location, and entites.
        '''
        rerun = []
        result = {}
        #OpenAI API Key
        for i in contest_number:
            # Getting the description 
            d1 = self.shots["image_description"].iloc[0]
            dud1 = self.shots["image_uncanny_description"].iloc[0]
            iL1 = self.shots["image_location"].iloc[0]
            str1 = ""
            for j in ast.literal_eval(self.shots["entities"].iloc[0].replace('\n', ',')):
                str1 += ((j.split("/"))[-1] + ", ")
            str1 = str1[:-2]

            d2 = self.shots["image_description"].iloc[1]
            dud2 = self.shots["image_uncanny_description"].iloc[1]
            iL2 = self.shots["image_location"].iloc[1]
            str2 = ""
            for j in ast.literal_eval(self.shots["entities"].iloc[1].replace('\n', ',')):
                str1 += ((j.split("/"))[-1] + ", ")
            str2 = str2[:-2]

            d3 = self.shots["image_description"].iloc[2]
            dud3 = self.shots["image_uncanny_description"].iloc[2]
            iL3 = self.shots["image_location"].iloc[2]
            str3 = ""
            for j in ast.literal_eval(self.shots["entities"].iloc[2].replace('\n', ',')):
                str1 += ((j.split("/"))[-1] + ", ")
            str3 = str3[:-2]

            d4 = self.shots["image_description"].iloc[3]
            dud4 = self.shots["image_uncanny_description"].iloc[3]
            iL4 = self.shots["image_location"].iloc[3]
            str4 = ""
            for j in ast.literal_eval(self.shots["entities"].iloc[3].replace('\n', ',')):
                str1 += ((j.split("/"))[-1] + ", ")
            str4 = str4[:-2]

            d5 = self.shots["image_description"].iloc[4]
            dud5 = self.shots["image_uncanny_description"].iloc[4]
            iL5 = self.shots["image_location"].iloc[4]
            str5 = ""
            for j in ast.literal_eval(self.shots["entities"].iloc[4].replace('\n', ',')):
                str1 += ((j.split("/"))[-1] + ", ")
            str5 = str5[:-2]

            #captions and label
            test1 = base64.b64encode(ast.literal_eval(self.shots["image"].iloc[0])['bytes']).decode("ascii")
            base64_image1 = test1

            test2 = base64.b64encode(ast.literal_eval(self.shots["image"].iloc[1])['bytes']).decode("ascii")
            base64_image2 = test2

            test3 = base64.b64encode(ast.literal_eval(self.shots["image"].iloc[2])['bytes']).decode("ascii")
            base64_image3 = test3

            test4 = base64.b64encode(ast.literal_eval(self.shots["image"].iloc[3])['bytes']).decode("ascii")
            base64_image4 = test4

            test5 = base64.b64encode(ast.literal_eval(self.shots["image"].iloc[4])['bytes']).decode("ascii")
            base64_image5 = test5


            #test6 = base64.b64encode(image_data.iloc[image_data['cnum'].tolist().index(i)]['image_bytes']['bytes']).decode("ascii")
            #base64_image6 = test6

            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

            payload = {
                        "model": "gpt-4o",
                        "messages": [
                            # 1st example
                            {"role": "user",
                             "content": [
                                {"type": "text", 
                                 "text": "In this task, you will see a cartoon, then write two descriptions about the cartoon, one uncanny description and one canny description, then write the cartoon's location, and the entities of the cartoon. I am going to give you five examples first and you write the last sets of description."
                                },        
                                {"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image1}",  "detail": "high"}
                                },
                             ]},
                            {"role": "assistant", 
                             "content": [{
                                "type": "text",
                                "text": "The canny description is "
                                + d1
                                + ", and the uncanny description is "
                                + dud1
                                + ", and the cartoon's location is "
                                + iL1
                                + ", and the entities of the cartoon are "
                                + str1[:-1]
                                + ".",
                                },
                             ]},
                            # 2 example
                             {"role": "user",
                             "content": [     
                                {"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image2}",  "detail": "high"}
                                },
                             ]},
                             {"role": "assistant", 
                             "content": [{
                                "type": "text",
                                "text": "The canny description is "
                                + d2
                                + ", and the uncanny description is "
                                + dud2
                                + ", and the cartoon's location is "
                                + iL2
                                + ", and the entities of the cartoon are "
                                + str2[:-1]
                                + ".",
                                },
                             ]},
                            # 3 example
                             {"role": "user",
                             "content": [     
                                {"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image3}",  "detail": "high"}
                                },
                             ]},
                             {"role": "assistant", 
                             "content": [{
                                "type": "text",
                                "text": "The canny description is "
                                + d3
                                + ", and the uncanny description is "
                                + dud3
                                + ", and the cartoon's location is "
                                + iL3
                                + ", and the entities of the cartoon are "
                                + str3[:-1]
                                + ".",
                                },
                             ]},
                            # 4 example
                             {"role": "user",
                             "content": [     
                                {"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image4}",  "detail": "high"}
                                },
                             ]},
                             {"role": "assistant", 
                             "content": [{
                                "type": "text",
                                "text": "The canny description is "
                                + d4
                                + ", and the uncanny description is "
                                + dud4
                                + ", and the cartoon's location is "
                                + iL4
                                + ", and the entities of the cartoon are "
                                + str4[:-1]
                                + ".",
                                },
                             ]},
                            # 5 example
                             {"role": "user",
                             "content": [     
                                {"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image5}",  "detail": "high"}
                                },
                             ]},
                             {"role": "assistant", 
                             "content": [{
                                "type": "text",
                                "text": "The canny description is "
                                + d5
                                + ", and the uncanny description is "
                                + dud5
                                + ", and the cartoon's location is "
                                + iL5
                                + ", and the entities of the cartoon are "
                                + str5[:-1]
                                + ".",
                                },
                             ]},
                            # 6 example
                            {"role": "user",
                            "content": [
                                {"type": "image_url",
                                 "image_url": {"url": 'https://nextml.github.io/caption-contest-data/cartoons/'+str(i)+'.jpg',  "detail": "high"}
                                },
                                {"type": "text",
                                "text": f"The set of description is "
                                }],
                            }
                        ],
                        "max_tokens": 1000,
                        "temperature": 0
                    }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
            )
            try:
                result[i] = response.json()["choices"][0]["message"]["content"]
            except:
                result[i] = "empty"
                rerun.append(i)

        # need to manual check and rerun
        canny  = []
        uncanny= []
        location = []
        entities = []
        cnum = []
        for i in result:
            cnum.append(i)
            try:
                canny.append(result[i].split("and the entities of the cartoon are")[0].split("and the cartoon's location is")[0].split("and the uncanny description is")[0].split("The canny description is ")[1][:-2])
                uncanny.append(result[i].split("and the entities of the cartoon are")[0].split("and the cartoon's location is")[0].split("and the uncanny description is")[1][:-2])
                location.append(result[i].split("and the entities of the cartoon are")[0].split("and the cartoon's location is")[1][1:-2])
                entities.append(result[i].split("and the entities of the cartoon are")[1].split(","))
            except:
                canny.append("")
                uncanny.append("")
                location.append("")
                entities.append("")
        d = {'cnum': cnum, 'canny': canny, "uncanny": uncanny, "location" : location, 'entities': entities}
        df = pd.DataFrame(data=d)
        return df
