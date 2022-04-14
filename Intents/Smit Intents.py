new_intents = [
    {
        "tag": " Greeting ",

        "patterns": [   
                        "Hi",
                        "Hi there",
                        "Hola",
                        "Hello",
                        "Hello there",
                        "Hya",
                        "Hya there "
                    ],

        "responses": [
                        " Hi, Sir/Ma'am happy to see you here. "
                     ],
        "context": [""]
    },
    {
        "tag": " GreetingResponse ",

        "patterns": [   
                        "My user is Sir/Ma'am",
                        "This is Sir/Ma'am",
                        "I am Sir/Ma'am",
                        "It is Sir/Ma'am"
                    ],

        "responses": [
                        "Great! Hi Sir/Ma'am! How can I help?",
                        "Good! Hi Sir/Ma'am, how can I help you?",
                        "Cool! Hello Sir/Ma'am, what can I do for you?",
                        "OK! Hola Sir/Ma'am, how can I help you?",
                        "OK! hi Sir/Ma'am, what can I do for you? "
                     ],
        "context": [" "]
    },
    {
        "tag": " Greetings  ",

        "patterns": [
                        " Good Morning",
                        " Good Afternoon",
                        " Good Evening "
                    ],

        "responses": [
                        " Good Morning Sir/Ma'am, what can I do for you?"
                        " Good Afternoon Sir/Ma'am, what can I do for you?"
                        " Good Evening Sir/Ma'am, what can I do for you? "
                    ],
        "context": [""]
    },
    {
        "tag": " Name ",

        "patterns": [
                        "What is your name?",
                        "What could I call you?",
                        "What can I call you?",
                        "What do your friends call you?",
                        "Who are you?",
                        "Tell me your name?"
                    ],

        "responses": [
                        "You can call me JARVIS",
                        "You may call me JARVIS",
                        "Call me JARVIS"
                     ],
        "context": [""]
    },
    {
        "tag": " RealName ",

        "patterns": [
                       "What is your real name?",
                        "What is your real name please?",
                        "What's your real name?",
                        "Tell me your real name?",
                        "Your real name?",
                        "Your real name please?",
                        "Your real name please?" 
                    ],

        "responses":[
                        "My name is JARVIS",
                        "JARVIS",
                        "My real name is JARVIS" 
                    ],
        "context": [""]
    },

    {
        "tag": " Thanks ",

        "patterns":[
                        "OK thank you",
                        "OK thanks",
                        "OK",
                        "Thanks",
                        "Thank you",
                        "That's helpful"
                    ],

        "responses":[
                        "No problem!",
                        "Happy to help!",
                        "Any time!",
                        "My pleasure"
                    ],
        "context": [""]
    },
    {
        "tag": " Shutup ",

        "patterns":[
                        "Be quiet",
                        "Shut up",
                        "Stop talking",
                        "Enough talking",
                        "Please be quiet",
                        "Quiet",
                        "Shhh"
                    ],

        "responses":[
                        "I am sorry to disturb you",
                        "Fine, sorry to disturb you",
                        "OK, sorry to disturb you"
                    ],
        "context": [""]
    },
    {
        "tag": " GoodBye ",

        "patterns":[
                        "Bye",
                        "Adios",
                        "See you later",
                        "Goodbye"
                    ],

        "responses":[
                        "See you later",
                        "Have a nice day",
                        "Bye! Come back again soon."
                    ],
        "context": [""]
    },
    {
        "tag": " CourtesyGoodBye ",

        "patterns":[
                        "Thanks, bye",
                        "Thanks for the help, goodbye",
                        "Thank you, bye",
                        "Thank you, goodbye",
                        "Thanks goodbye",
                        "Thanks good bye"
                    ],

        "responses":[
                        "No problem, goodbye",
                        "Not a problem! Have a nice day",
                        "Bye! Come back again soon."
                    ],
        "context": [""]
    },
    {
        "tag": "Shirts_Types ",

        "patterns":[
                        "What kind of shirts do you have",
                        "What are the different varieties of shirts you provide",
                        "Do you have half sleves shirts?",
                        "Do you have full sleves shirts?"
                    ],

        "responses":[
                        "We provide a wide variety of Shirts which includes - Full Sleves Shirts, Half Sleves Shirts, Plain Shirts, Printed Shirts and many more."
                    ],
        "context": [""]
    },
    {
        "tag": "Shirt_Categories ",

        "patterns":[
                        "Do you have Plain Shirts?",
                        "Do you have Printed Shirts?",
                        "Do you have Half-Sleves Shirts?",
                        "Do you have Full-Sleves Shirts?",
                    ],

        "responses":[
                        "The different categories of the Shirts which includes are - Full Sleves Shirts, Half Sleves Shirts, Plain Shirts, Printed Shirts and many more."
                    ],
        "context": [""]
    },
    {
        "tag": " Shirt_Sizes ",

        "patterns":[
                        "What are the sizes of shirts do you have",
                        "What are the different shirt sizes you provide",
                        "What are the sizes for half selves shirts?",
                        "What are the sizes for half selves shirts?"
                    ],

        "responses":[
                        "Shirt Size Chart for Men"
                    ],
        "context": [""]
    },
    {
        "tag": "Shirt_Colours ",

        "patterns":[
                        "What kind of colours of shirts do you provide",
                        "What are the different varieties of colours of shirts do you provide?",
                        "Do you provide multi-colour shirts?"
                    ],

        "responses":[
                        "Yes we provide multi-colour shirts with a vast varieties of colours. Some of the colours of the shirts are white, black, blue, navy blue, and mnay more coloured shirts."             
                    ],
        "context": [""]
    },
    {
        "tag": "Shirt_Prices",

        "patterns":[
                        "What is the price range for the shirts?",
                        "Do you have a very high price range of shirts?"
                    ],

        "responses":[
                        "The price range for Shirts is from 500-2000 INR."
                    ],
        "context": [""]
    },
    {
        "tag": " ",

        "patterns":[
                        ""
                    ],

        "responses":[

                    ],
        "context": [""]
    },
    {
        "tag": " ",

        "patterns":[
                        ""
                    ],

        "responses":[

                    ],
        "context": [""]
    },
    {
        "tag": " ",

        "patterns":[
                        ""
                    ],

        "responses":[

                    ],
        "context": [""]
    },
    {
        "tag": " ",

        "patterns":[
                        ""
                    ],

        "responses":[

                    ],
        "context": [""]
    },
    {
        "tag": " ",

        "patterns":[
                        ""
                    ],

        "responses":[

                    ],
        "context": [""]
    },
    {
        "tag": " ",

        "patterns":[
                        ""
                    ],

        "responses":[

                    ],
        "context": [""]
    },
    {
        "tag": " ",

        "patterns":[
                        ""
                    ],

        "responses":[

                    ],
        "context": [""]
    },
    {
        "tag": " ",

        "patterns":[
                        ""
                    ],

        "responses":[

                    ],
        "context": [""]
    },
    {
        "tag": " ",

        "patterns":[
                        ""
                    ],

        "responses":[

                    ],
        "context": [""]
    },
    {
        "tag": " ",

        "patterns":[
                        ""
                    ],

        "responses":[

                    ],
        "context": [""]
    },
    
]   