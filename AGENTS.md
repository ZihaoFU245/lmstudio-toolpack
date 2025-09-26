WebSearch.py is a searching tool for AI Agents. (Your target)

Some websites like reddit, is blocking my tool reaching it.
Add more details, a real browser without login can still visit. 

Rewrite tool "visit_url" that extract all text from the web page including a BFS. Apply AD blocking. 

The response would be 
```
[1] Title:
    URL: https://example.com
    Referer: On BFS, state where is this url comming from. None if it is the input url.
    Detail: Contents on the page
```
Contents still need to be filtered. Keep all text and some <a> links. Filter out if the link is
an ad. 

Enforce the above new standard to "web_search" tool. 
Remove Excerpt, replace to detail. Apply the same rule. 

1. Trancation needed. 
Each result would have an upper limit of return. If the result is too long,
truncate it and add "The response is too long that is truncated, further reading please use visit_url tool for {{ url_traget }}"

Summary:
1. Rewrote the tools
2. Enforce Browser simulation
3. Replace Excerpt to Detail
4. Prune unnecessary code if needed