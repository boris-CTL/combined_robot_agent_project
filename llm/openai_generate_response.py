import openai
import time
import traceback

def openai_chat_response(messages, temperature = 1.0, gpt="gpt-3.5-turbo"): #gpt="gpt-4"):
    for i in range(10):
      # check = input("Do you want to send these messages to GPT-4? (y/n)\n{}".format(messages))
      # if check != 'y':
      #   return None
      print(f'Prompt:\n{messages}')
      # input('Continue?\n')

      try:
        client = openai.OpenAI(organization='org-VFSaZQgJUGtyglfUzEzesHAC')

        response = client.chat.completions.create(
        model=gpt,
        messages=messages,
        temperature=temperature
        )
        return response.choices[0].message.content
      except:
        traceback.print_exc()
        time.sleep(i*100)