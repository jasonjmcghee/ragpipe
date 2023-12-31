This is being used for testing related to [rem](https://github.com/jasonjmcghee/rem)

# Used like:

```bash
$ ./askRem "Which GitHub issues have I read recently?" <(sqlite3 db 'select text from allText order by frameId desc limit 1000') 
Batches: 100%|███████████████████████████████| 19/19 [00:11<00:00,  1.65it/s]
```

```md
You have recently read issues: #3 (dark mode icons), #9 (login item - Rem will run on boot), and #11 (icon looks kinda weird when active in dark mode).
```

```md
total duration:       26.622822625s
load duration:        5.327591125s
prompt eval count:    1933 token(s)
prompt eval duration: 17.73078s
prompt eval rate:     109.02 tokens/s
eval count:           41 token(s)
eval duration:        3.554184s
eval rate:            11.54 tokens/s
```
