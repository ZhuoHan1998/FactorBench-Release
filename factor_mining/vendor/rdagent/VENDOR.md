
## Note on nested .gitignore

The upstream repository's own `.gitignore` was removed from this snapshot:
inside a vendored tree it makes the parent repo silently drop real source
files from commits (this bit us three times). No other file is affected.
