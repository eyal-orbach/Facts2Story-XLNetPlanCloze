from transformers import GPT2Config, GPT2Tokenizer
import json


path1="/home/nlp/eyalo/data/movie_plots/salie_plots_tokens_xlnet_details//train/train_18097.json"
path2="/home/nlp/eyalo/data/movie_plots/salie_plots_tokens_xlnet_details//train/train_1965.json"

target="/home/nlp/eyalo/data/movie_plots/blind_test_plots/final/76.json"
out_path="/home/nlp/eyalo/data/movie_plots/gpt3_prompt.txt"

config = GPT2Config.from_pretrained('gpt2')
gpt2tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def get_facts(script_json):
    facts_str=""
    i=1
    for fact in script_json["openfacts"]:
        facts_str+=f"Fact {i}: " + fact["text"]+". "
        i+=1
    return facts_str

def get_story(script_json):
    story_str="Story: " + script_json["text"]
    return story_str


if __name__ == '__main__':
    prompt=""
    with open(path1) as f1:
        story1 = json.load(f1)
        prompt+= get_facts(story1)
        prompt+= get_story(story1)
        # prompt+="<sep>"
    with open(path2) as f2:
        story2 = json.load(f2)
        prompt+= get_facts(story2)
        prompt+= get_story(story2)
        # prompt+="<sep>"
    with open(target) as f3:
        story3 = json.load(f3)
        prompt+= get_facts(story3)
        prompt+= "Story:"

    tokens = gpt2tokenizer.tokenize(prompt)
    print("num tokens:" + str(len(tokens)))
    with open(out_path, "w") as out:
        out.write(prompt)