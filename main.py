import json

l='''I got to go now
I gotta go
I'll be back
I'll be back soon
I'll get back to you
I'll miss you
It's been nice to talk to you
adios
alright bye
alright good night
appreciate the chat
asta la vista
back in a bit
be back in 5 minutes
be back in a few
bye
bye bye 
bye bye see you
bye bye see you soon
bye bye take care
bye for now
bye good night
bye-bye
chat later
cheerio
cheers
ciao
fine for now
get lost
go to bed
going to bed
going to bed now
good bye
good night
good night bye
good night to you
good talking to you
good to chat
goodbye
goodbye for now
goodbye see you later
goodnight now
got get sleep
gotta go to sleep
have a good night
hope to see you later
it's bed time
it's been a pleasure chatting with you
later you
leave me alone
nice talking to you
nice to chat
nice to talk to you
ok bye
ok have a good night
okay bye
okay see you later
okay thank you bye
sayonara
see ya
see you
see you soon
see you tomorrow
speak to ya
sweet dreams
ta ta for now
take care
take it easy
talk to you later
thanks bye
thanks bye bye
thanks for chatting
thanks good night
that's all for now
til next time
til we meet again
till next time
time to go
time to go to bed
you can go now'''
k=[]
print(json.dumps(l.split("\n")))
