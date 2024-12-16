from bardapi import Bard

token = 'cwjYKur_l-s9bp3wzoP31ZA9JQk68d_fBE6xxo9MXKUdsqZSamP_YC-YGbr7UH4rTO7QWQ.'
bard = Bard(token=token)
print(bard.get_answer("How are you?")['content'])