# minimal-ir

Minimalistic information retrieval system based on the vector space modeling. The system recommends documents included in *corpus* to users in *profiles* based on the interests of each user.

```text
User1#movies#politics
User2#politics#soccer
User3#politics
```

For example, it will recommend documents about movies and politics to the first user. You can add entries to this file to include more user profiles. However, for the moment only 4 topics are supported: `movies`, `politics`, `soccer` and `books` (the dictionary only includes these terms).

Run the program with the following command:

```bash
python profileir.py
```

You can also run the tests:

```bash
python -m unittest discover
```

This should be the output:

```text
******************************************
   Terms frequencies (similar grouped)    
******************************************
{'blade-runner': {'movi': 8, 'politic': 0, 'soccer': 0},
 'chelsea': {'movi': 0, 'politic': 0, 'soccer': 13},
 'film-quiz': {'movi': 6, 'politic': 0, 'soccer': 0},
 'labour-activist': {'movi': 0, 'politic': 5, 'soccer': 0},
 'sevilla-coach': {'movi': 0, 'politic': 0, 'soccer': 8},
 'voters-ID-plan': {'movi': 0, 'politic': 8, 'soccer': 0}}
******************************************
==========================================
                  User1                   
==========================================
       Interests: movies & politics       
==========================================
   Recommendation   ||       Score        
==========================================
    blade-runner    ||   0.831890330808   
==========================================
     film-quiz      ||   0.471404520791   
==========================================
   voters-ID-plan   ||   0.441941738242   
==========================================
  labour-activist   ||   0.304787405684   
==========================================
==========================================
                  User2                   
==========================================
       Interests: politics & soccer       
==========================================
   Recommendation   ||       Score        
==========================================
      chelsea       ||   0.638360288571   
==========================================
   sevilla-coach    ||   0.614875461901   
==========================================
   voters-ID-plan   ||   0.441941738242   
==========================================
  labour-activist   ||   0.304787405684   
==========================================
==========================================
                  User3                   
==========================================
           Interests: politics            
==========================================
   Recommendation   ||       Score        
==========================================
   voters-ID-plan   ||       0.625        
==========================================
  labour-activist   ||   0.431034482759   
==========================================
Documents with score less than 0.1 are hidden
```

You can also add more documents to the *corpus* directory.
