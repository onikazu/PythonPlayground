import MeCab

print("start")

text = '自家発電機'
m = MeCab.Tagger()
s = m.parse(text)
print(s)