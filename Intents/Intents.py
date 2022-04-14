new_intents = [
    {
        "tag": " Greeting ",

        "patterns": [   
                        "Hi",
                        "Hi there",
                        "Hey",
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
        "tag": "About",

        "patterns":[
                        "Tell me about yourself ?",
                        "Tell me about your self ?",
                        "What do you sell?",
                        "What the chatbot about?",
                        "What do you have?",
                        "What is this about",
                        "what do you provide"
                    ],

        "responses": [
                        "We at fashion factory have clothes for Men and Women. For Men we have T-shirts, Shirts, Pants, Jeans, Trousers, Shoes,etc. For Women we have Shirts, Jeggings, Tops, Kurtis, Dress-Materials, Sleepwear, etc. "
                     ],

        "context": [" "]
    },
    {
        "tag": " GreetingResponse ",

        "patterns": [   
                        "My name is Sir/Ma'am",
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
        "tag": " Greetings_GM ",

        "patterns": [
                        " Good Morning",
                        " GM"
                    ],

        "responses": [
                        " Good Morning Sir/Ma'am, what can I do for you?"
                    ],
        "context": [""]
    },
    {
        "tag": " Greetings_GA ",

        "patterns": [
                        " Good Afternoon",
                        " GA "
                    ],

        "responses": [

                        " Good Afternoon Sir/Ma'am, what can I do for you?"
                    ],
        "context": [""]
    },
    {
        "tag": " Greetings_GE ",

        "patterns": [
                        " GE",
                        " Good Evening "
                    ],

        "responses": [

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
                        "Tell me your name?",
                        "Hello, chatbot my name is Sir/Ma'am what is you name"
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
                        "What can I call you?" 
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
        "tag": "Shirts",

        "patterns":[
                        "Shirts ",
                        "Plain Shirts",
                        "Printed Shirts",
                        "Half sleeves Shirts",
                        "Formal Shirts",
                        "Polo Shirts"
                    ],

        "responses": [
                        "Yes we provide these types of shirts like Plain Shirts, Printed Shirts, Half sleeves Shirts, Formal Shirts,Polo Shirts and many more. "
                     ],

        "context": [" "]
    },
    {
        "tag": "Shirts_Types ",

        "patterns":[
                        "Types of Shirts",
                        "What kind of shirts do you have",
                        "What are the different varieties of shirts you provide",
                        "Do you have half sleeves shirts?",
                        "Do you have full sleeves shirts?"
                    ],

        "responses":[
                        "We provide a wide variety of Shirts which includes - Full Sleeves Shirts, Half Sleeves Shirts, Plain Shirts, Printed Shirts and many more."
                    ],
        "context": [""]
    },
    {
        "tag": "Shirt_Categories ",

        "patterns":[
                        "Categories in Shirts",
                        "Do you have Plain Shirts?",
                        "Do you have Printed Shirts?",
                        "Do you have Half-sleeves Shirts?",
                        "Do you have Full-sleeves Shirts?",
                        "Do you provide Plain Shirts?",
                        "Do you provide Printed Shirts?",
                        "Do you provide Half-sleeves Shirts?",
                        "Do you provide Full-sleeves Shirts?"
                    ],

        "responses":[
                        "We provide different kinds of Shirts like - Full Sleeves Shirts, Half Sleeves Shirts, Plain Shirts, Printed Shirts and many more."
                    ],
        "context": [""]
    },
    {
        "tag": "Shirt_Categories_Plain_Shirts ",

        "patterns":[
                        "Do you have Plain Shirts?",
                        "Do you provide Plain Shirts?",
                        "Plain Shirts",
                        "Simple Plain  Shirts",
                    ],

        "responses":[
                        "Yes we have Plain Shirts."
                    ],
        "context": [""]
    },
    {
        "tag": "Shirt_Categories_Printed_Shirts ",

        "patterns":[
                        "Do you have Printed Shirts?",
                        "Do you provide Printed Shirts?",
                        "Printed Shirts",
                        "Simple Printed  Shirts",
                    ],

        "responses":[
                        "Yes we provide Printed Shirts."
                    ],
        "context": [""]
    },
    {
        "tag": "Shirt_Categories_Sleeves_Shirts ",

        "patterns":[
                        "Do you have Half Sleeves Shirts?",
                        "Do you have Full Sleeves Shirts?",
                        "Do you provide Half Sleeves Shirts?",
                        "Do you provide Full Sleeves Shirts?",
                        "Half Sleeves Shirts",
                        "Full Sleeves Shirts"
                    ],

        "responses":[
                        "Yes we provide Half Sleeves Shirts as well as Full Sleeves Shirts",
                        "Yes we provide Full Sleeves Shirts as well as Half Sleeves Shirts"
                    ],
        "context": [""]
    },
    {
        "tag": "Shirt_Categories_Formal_Shirts ",

        "patterns":[
                        "Do you have Formal Shirts?",
                        "Do you provide Formal Shirts?",
                        "Formal Shirts",
                        "Simple Formal Shirts",
                        "Casual Formal Shirts",
                        "Party Formal Shirts",
                        "Party-Casual Formal Shirts"
                    ],

        "responses":[
                        "Yes we provide Formal Shirts."
                    ],
        "context": [""]
    },
    {
        "tag": "Shirt_Categories_Polo_Shirts ",

        "patterns":[
                        "Do you have Polo Shirts?",
                        "Do you provide Polo Shirts?",
                        "Polo Shirts",
                        "Simple Polo  Shirts",
                    ],

        "responses":[
                        "Yes we provide Polo Shirts."
                    ],
        "context": [""]
    },
    {
        "tag": " Shirt_Sizes ",

        "patterns":[
                        "Size for Shirts",
                        "What are the sizes of shirts do you have",
                        "What are the different shirt sizes you provide",
                        "What are the sizes for half selves shirts?",
                        "What are the sizes for half selves shirts?"
                    ],

        "responses":[
                        "Shirt Size Chart for Men. We provide Sizes from XS-XXL."
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
                        "Do you have a very high price range of shirts?",
                        "Price range for Shirts",
                        "Prices for Shirts",
                        "What is the price of shirts?",
                        "What is the average price of shirts",
                        "What is the basic or normal price range for shirts"
                    ],

        "responses":[
                        "The price range for Shirts is from 500-2000 INR."
                    ],
        "context": [""]
    },
    {
        "tag": "T-Shirts",

        "patterns":[
                        "T-Shirts ",
                        "Plain T-Shirts",
                        "Printed T-Shirts",
                        "Half sleeves T-Shirts",
                        "Formal T-Shirts",
                        "Polo T-Shirts"
                    ],

        "responses": [
                        "Yes we provide these types of shirts like Plain T-Shirts, Printed T-Shirts, Half sleeves T-Shirts, Formal T-Shirts,Polo T-Shirts and many more. "
                     ],

        "context": [" "]
    },
    {
        "tag": "T-Shirt_Types",

        "patterns":[
                        "What kind of t-shirts do you have",
                        "What are the different varieties of t-shirts you provide",
                        "Types in T-shirts",
                        "What are the different Types in T-shirts"
                    ],

        "responses": [
                        "The different types of the T-Shirts which includes are - Full sleeves t-Shirts, Half sleeves t-Shirts, Plain t-Shirts, Polo t-Shirts, round-neck t-shirts, V-neck T-rshirts and many more."
                     ],

        "context": [" "]
    },
    {
        "tag": "T-Shirt_Categories",

        "patterns":[
                        "What are the different categories in T-shirts? ",
                        "Which categories in t-shirts do you provide?",
                        "Categories in T-shirts",
                        "T-shirt categories",
                        "Do you have Plain T-Shirts?",
                        "Do you have Printed T-Shirts?",
                        "Do you have Half-sleeves T-Shirts?",
                        "Do you have Full-sleeves T-Shirts?",
                        "Do you provide Plain T-Shirts?",
                        "Do you provide Printed T-Shirts?",
                        "Do you provide Half-sleeves T-Shirts?",
                        "Do you provide Full-sleeves T-Shirts?",
                        "Do you provide V-neck T-Shirts?",
                        "Do you have V-neck T-Shirts?"
                    ],

        "responses": [
                        " Their are different categories in T-shirts such as Full sleeves t-shirts, Half sleeves t-Shirts, Plain t-Shirts, Polo t-Shirts, round-neck t-shirts, V-neck T-rshirts and many more."
                     ],

        "context": [" "]
    },
    {
        "tag": "T-Shirt_sizes",

        "patterns":[
                        " ",
                        " "
                    ],

        "responses": [
                        " "
                     ],

        "context": [" "]
    },
    {
        "tag": "T-Shirt_Colours",

        "patterns":[
                        " ",
                        " "
                    ],

        "responses": [
                        " "
                     ],

        "context": [" "]
    },
    {
        "tag": "T-Shirt_Price",

        "patterns":[
                        "What is the price range for the t-shirts?",
                        "Do you have a very high price range of t-shirts?"
                    ],

        "responses":[
                        "The price range for T-Shirts is from 250-1000 INR."
                    ],
        "context": [""]
    },
    {
        "tag": "",

        "patterns":[
                        " ",
                        " "
                    ],

        "responses": [
                        " "
                     ],

        "context": [" "]
    },
    {
        "tag": "",

        "patterns":[
                        " ",
                        " "
                    ],

        "responses": [
                        " "
                     ],

        "context": [" "]
    },
    {
        "tag": "",

        "patterns":[
                        " ",
                        " "
                    ],

        "responses": [
                        " "
                     ],

        "context": [" "]
    },
    {
        "tag": "",

        "patterns":[
                        " ",
                        " "
                    ],

        "responses": [
                        " "
                     ],

        "context": [" "]
    },
    {
        "tag": "",

        "patterns":[
                        " ",
                        " "
                    ],

        "responses": [
                        " "
                     ],

        "context": [" "]
    },
    {
        "tag": "",

        "patterns":[
                        " ",
                        " "
                    ],

        "responses": [
                        " "
                     ],

        "context": [" "]
    },
    {
        "tag": "",

        "patterns":[
                        " ",
                        " "
                    ],

        "responses": [
                        " "
                     ],

        "context": [" "]
    },
    {
        "tag": "",

        "patterns":[
                        " ",
                        " "
                    ],

        "responses": [
                        " "
                     ],

        "context": [" "]
    },
    {
        "tag": "",

        "patterns":[
                        " ",
                        " "
                    ],

        "responses": [
                        " "
                     ],

        "context": [" "]
    },
    {
        "tag": "",

        "patterns":[
                        " ",
                        " "
                    ],

        "responses": [
                        " "
                     ],

        "context": [" "]
    },
    















    {
        "tag": "women_sleepwear",

        "patterns": [
                        "sleep wear for women", 
                        "night dresses for women", 
                        "I want night dress for women", 
                        "show me night dress for ladies", 
                        "ladies night dress", 
                        "night dress for girls", 
                        "women night dresses", 
                        "sleep wear for girls", 
                        "I want to see sleep wear for women",
                        "ladies night wear",
                        "women night wear", 
                        "what material of night dresses are available for women",
                        "fabrics available for girls night dresses"
                    ],

        "responses": ["Sure, we provide a wide range and variety of sleep wear ranging from pajamas, sleeping gowns, sleepers, nightshirts to nighties and night robes for women and girls. All types of fabrics like cotton, velvet, linen, silk and many are available. Check the women's sleep and lounge wear section for latest trendy outfits and offers." ],
        
        "context": [""]
    },

    {
        "tag": "women_sleepwearsize",

        "patterns": [
                        "size for women night dress", 
                        "size for women sleep wear",
                        "Show me sizes for women night wear",
                        "I want to see different sizes of women's sleep wear",
                        "What all sizes do you offer for ladies night dresses?",
                        "girls night dress size",
                        "size of girls night wear",
                        "size of ladies night wear"
                    ],

    "responses": ["We have all size ranges available here at Fashion Factory. They range from XS, S, M, L, XL, 2XL to 3XL. Checkout the women's sleep and lounge wear section for latest trendy outfits and offers."], 
    },

    {
        "tag": "women_sleepwearprice",

        "patterns": [
                        "Show me the price range for sleep wear of girls", 
                        "price range for women sleep wear", 
                        "range for ladies sleep wear",
                        "show me the rates of ladies sleep wear",
                        "rate of women sleep wear",
                        "prices of ladies night dress",
                        "I want to check prices of night dresses for women",
                        "night dress price for women",
                        "ladies night dress rates"
                    ],

        "responses": ["The price ranges from 199 to 5999. Checkout the women's sleep and lounge wear section for latest trendy outfits and offers."],
    },

    {
        "tag": "women_sleepweartypes",

        "patterns": [
                        "Show me the types of sleep wear of girls", 
                        "types of women sleep wear", 
                        "types of ladies sleep wear",
                        "show me the different types of ladies sleep wear",
                        "types of women sleep wear",
                        "different types of ladies night dress",
                        "I want to check different types of night dresses for women",
                        "night dress varieties for women",
                        "ladies night dress varieties",
                        "I want to see all varieties of women sleep wear that you provide",
                        "Show me all the varieties of sleep wear that you provide"
                    ],

        "responses": ["We provide various types and varieties of sleep wear including Sleep Shirt/Night Shirt, Bathrobes/Robe Sets, Night Dress, Playsuit/Jumpsuit, Nightie/Nightgown, Nightwear Jumpsuit, Night Kaftans, Nighty Slip, Full sleeve Night wear, Two-piece, Three-piece, Five-piece Nightwears, Pyjama Set, Shorts Set, Sleep Shorts for Teens, One-piece Nightwear, Maternity Night wear, Pair night wears, Capri Set and Babydoll. Checkout the women's sleep and lounge wear section for latest trendy outfits and offers."],
    },

    {
        "tag": "women_loungewear",

        "patterns": [
                        "lounge wear for women", 
                        "pajamas for women", 
                        "I want home wear for women", 
                        "show me pajama suits for ladies", 
                        "ladies pajama suits", 
                        "lounge wear and home wear", 
                        "lounge wear for girls", 
                        "women pajama dresses", 
                        "lounge wear for girls", 
                        "I want to see home wear for women", 
                        
                    ],

        "responses": ["we provide a wide range and variety of sleep wear ranging from comfortable t-shirt sets, sweatpants, pajamas, sleepers, nightshirts to nighties for women and girls. All types of fabrics like cotton, velvet, linen, silk and many more are available. Check the women's sleep and lounge wear section for latest trendy outfits."],

        "context": [""]
    },

    {
        "tag": "women_loungewearsize",

        "patterns": [
                        "size for ladies lounge wear",
                        "Show me size range for ladies lounge wear",
                        "size for women lounge dress", 
                        "size for women home wear",
                        "Show me sizes for women home wear",
                        "I want to see different sizes of women's lounge wear",
                        "What all sizes do you offer for ladies home dresses?",
                        "girls lounge dress size",
                        "size of girls home wear",
                        "size of ladies lounge wear"
                    ],

        "responses": ["We have all size ranges available here at Fashion Factory. They range from XS, S, M, L, XL, 2XL to 3XL. Checkout the women's sleep and lounge wear section for latest trendy outfits and offers."],

        "context": [""]
    },

    {
        "tag": "women_loungewearprice",

        "patterns": [
                        "Show me the price range for lounge wear of girls", 
                        "price range for women lounge wear", 
                        "range for ladies lounge wear", 
                        "price range for women lounge wear", 
                        "range for ladies home wear",
                        "show me the rates of ladies lounge wear",
                        "rate of women home wear",
                        "prices of ladies home dress",
                        "I want to check prices of lounge dresses for women",
                        "pajama dress price for women",
                        "ladies home dress rates",
                        "Show me pajama dress rates for women"
                    ],

        "responses": ["The price ranges from 299 to 6999. Checkout the women's sleep and lounge wear section for latest trendy outfits and offers."],

        "context": [""]
    },

    {
        "tag": "women_loungeweartypes",

        "patterns": [],

        "responses": ["We provide various types and varieties of sleep wear including Tank and shorts, Thermal Underwear, Sweats, Tracksuits, Leggings, Jumpsuits, Slacks and T-Shirts, Relaxed pants, Cashmere Hoodie, Pajamas, Onesies, Playsuits, Jumpsuits, Camisoles, Maxi dresses and Capri sets. Checkout the women's sleep and lounge wear section for latest trendy outfits and offers."],

        "context": [""]
    },

    {
        "tag": "women_sportswear",

        "patterns": [
                        "sports wear for women", 
                        "active wear for women", 
                        "I want sports dress for women", 
                        "show me sports leggings for ladies", 
                        "ladies athletic wear", 
                        "active wear and sports wear", 
                        "athletic wear for girls", 
                        "women gym wear", 
                        "sports wear for girls",
                        "I want to see gym wear for women", 
                        
                        
                    ],

        "responses": ["Surely, we provide a wide variety of sports wear that can be utilised for exercise purposes as well as particular sports. Our products range from yoga pants, tights, sneakers, leggings and shorts for women and girls. All types of fabrics like polyester, spandex, neoprene and many more are available. They're durable, wrinkle-resistant, lightweight, and non-absorbent, which means that moisture from your skin evaporates instead of being drawn into the material. The price ranges from 599 to 6999. Also, all sizes between XS to 3XL are available. Check the women's Sports and Active wear section for latest trendy outfits."],

        "context": [""]
    },

    {
        "tag": "women_sportswearsize",

        "patterns": [
                        "size for women exercise wear",
                        "size for ladies active wear",
                        "Show me size range for ladies gym and sports wear",
                        "size for women sports dress", 
                        "size for women exercise wear",
                        "Show me sizes for women sports wear",
                        "I want to see different sizes of women's gym wear",
                        "What all sizes do you offer for ladies active wear dresses?",
                        "girls gym dress size",
                        "size of girls sports wear",
                        "size of ladies active wear",
                        "I want to checkout all sizes of exercise wear for women"
                    ],

        "responses": ["We have all size ranges available here at Fashion Factory. They range from XS, S, M, L, XL, 2XL to 3XL. Checkout the women's Sports and Active wear section for latest trendy outfits and offers."],

        "context": [""]
    },

    {
        "tag": "women_sportswearprice",

        "patterns": [
                        "Show me the price range for sports wear of girls", 
                        "price range for women exercise dress", 
                        "range for ladies athletic wear",
                        "range for ladies exercise wear",
                        "show me the rates of ladies gym wear",
                        "rate of women exercise dress",
                        "prices of ladies active wear",
                        "I want to check prices of sports dresses for women",
                        "gym wear price for women",
                        "ladies active wear dress rates",
                        "Show me sports dress rates for women",
                        "What are the price ranges of women's sports and active wear?"
                    ],

        "responses": ["The price ranges from 349 to 5499. Checkout the women's Sports and Active wear section for latest trendy outfits and offers."],

        "context": [""]
    },

    {
        "tag": "women_sportsweartypes",

        "patterns": [],

        "responses": ["We provide different types and varieties of sports wear like Tops, Tank Tops, Long-Sleeved Shirts, Racerbacks, Pants, Sweat Pants, Sweat Shorts, Bermuda Shorts, Leggings, Capri Pants, Socks, Low Cut Socks, Running Socks, Elite Micro Socks, Compression Socks, Footwear, Running Shoes, Minimalist Shoes, Innerwear, Sports Bras and much more. Checkout the women's Sports and Active wear section for latest trendy outfits and offers."],

        "context": [""]
    },





    {
        "tag": "women_purses",

        "patterns": [
                        "Show me handbags for women", 
                        "women handbags", 
                        "ladies clutch", 
                        "purse for ladies", 
                        "Show me purses for women", 
                        "fancy purses for girls", 
                        "latest purses", 
                        "Do you have handbags for women?", 
                        "Show me all categories of ladies purses", 
                        "handbags for girls", 
                        "clutches for women", 
                        "Classic purses and handbags", 
                        "girls handbags", 
                        "sling bags for women", 
                        "sling bags for girls"
                    ],

        "responses": ["Yes, we have a great variety of handbags, sling bags, clutches for women ranging from tote bags, shoulder bags, saddle cross-body bags, satchel bags, hobo bags, barrel bags to belt bags. Also, for professional purposes we provide laptop bags, duffel bags, messenger bags etc. For fancy occasions there are clutches, envelop bags and many more. Also, we provide latest brands and amazing deals. For more information, click on the Accessories section and check it out."],

        "context": [""]
    },

    {
        "tag": "women_pursesprice",

        "patterns": [
                        "show me sling bag prices",
                        "price ranges for women's purses",
                        "price for women's handbags",
                        "rates for ladies purses",
                        "ladies purse prices",
                        "ladies handbags rates",
                        "price of sling bags",
                        "Show me price ranges of handbags"
                    ],

        "responses": ["The price ranges are from 599 to 9999. Also, we provide latest brands and amazing deals. For more information, click on the Accessories section and check it out."],

        "context": [""]
    },

    {
        "tag": "women_rings",

        "patterns": [
                        "show me rings for women", 
                        "girls rings", 
                        "rings for ladies", 
                        "Do you provide rings for women?", 
                        "show rings for girls"
                    ],

        "responses": ["We provide trendy rings and you can check them out in Accessories section."],

        "context": [""]
    },

    {
        "tag": "women_bracelets",

        "patterns": [   
                        "show me bracelets for women", 
                        "girls bracelets", 
                        "bracelets for ladies", 
                        "do you provide bracelets for women", 
                        "show bracelets for girls", 
                        "ladies bracelets"
                    ],

        "responses": ["We provide trendy bracelets and you can check them out in Accessories section for women."],

        "context": [""]
    },
    {
        "tag": "Women Section ",

        "patterns":[
                        "What all does Women Section include?",
                       " I want to see different Categories of women section?",
                       "What is there in women section?",
                       "Women section categories?",
                       "Show me varieties available in women section?",
                    ],

        "responses":[
                        "We provide a wide variety which includes Shirts which includes Ethnic Wear,Western Wear,Sleep and Lounge Wear,Sportswear,Accessorie."
                    ],
        "context": [""]
    },
    {
        "tag": "Ethnic_Wear ",

        "patterns":[
                        "What are the different varieties available in Ethnic Wear?",
                        "What all do you have?",
                        "What all does ethnic wear include?",
                        "Do you have Sarees?",
                        "Is Saree Available?",
                        "Do you have Kurtas?",
                         "Is Kurta Available?",
                        "Do you have Salwar Suits?",
                         "Is Salwar Suit Available?",
                        "Do you have Dress Material?",
                         "Is Dress Material Available?",
                        "Do you have Bottom wear?",
                         "Is Bottom Available?",
                        "Do you have Dupattas?",
                         "Is Dupattas Available?",
                        "Do you have Bottoms?",
                        "Do you have Blouses?",
                    ],

        "responses":[
                        "We provide a wide variety which includes Sarees,Kurtas,Salwar Suits,Dress Material,Bottom wear,Dupattas,Bottoms,Blouses."
                    ],
        "context": [""]
    },
    {
        "tag": "Western_Wear ",

        "patterns":[
                        "What are the different varieties available in Western Wear?",
                        "What do you have in Western Wear?",
                        "Show all Western Wear variety?,"
                        "Do you have Tops?",
                        "Are tops available?",
                        "Do you have T-shirts?",
                        "Are T-shirts available?",
                        "Do you have Dresses?",
                        "Are Dresses available?",
                         "Do you have Jumpsuits?",
                        "Are Jumpsuits available?",                      
                         "Do you have Skirts?",
                        "Are Skirts available?",
                         "Do you have Jeans & Jeggings?",
                        "Are Jeans & Jeggings available?",
                         "Do you have Winterwear?",
                        "Are winterwear available?",
                        "Do you have Trousers?",
                        "Are Trousers available?",
                        "Do you have Leggings?",
                        "Are Leggings available?",
                        "Do you have Suits & Blazers?",
                        "Are Suits & Blazers available?",
                        

                    ],

        "responses":[
                        "We provide a wide variety which includes Western Wear Tops, T-Shirts & Shirts,Dresses & Jumpsuits,Skirts & Shorts,Jeans & Jeggings,Winterwear,Trousers,Leggings,Ponchos,Ponchos & Capes,Rainwear,Suits & Blazers."
                    ],
        "context": [""]
    },
    {
        "tag": "Sleep & Lounge Wear ",

        "patterns":[
                        "What are the different varieties available in Sleep & Lounge Wear?",
                        "What all do you have in Sleep & Lounge Wear?",
                        "Show all Sleep & Lounge Wear"
                        "Do you have Lounge shorts",
                        "Are Lounge shorts available?"
                        "Do you have Pyjama Sets",
                        "Are Pyjama available"
                        "Do you have nightwear sets"
                        "Are nightwear sets available"

                    ],

        "responses":[
                        "We provide a wide variety which includes Nighties & Nightdresses,Pyjama Sets,Pyjamas & Lounge Pants,Babydolls,Nightwear Sets,Lounge Shorts."
                    ],
        "context": [""]
    },
    {
        "tag": "Sportswear ",

        "patterns":[
                        "What are the different varieties available in Sportswear?",
                         "What all do you have in Sportswear?",
                         "Show all sportswear?"
                        "Do you have Sweatshirts & Hoodies",
                        "Are Sweatshirts & Hoodies available?"
                        "Do you have Trousers",
                        "Are Trousers available"
                        "Do you have Leggings"
                        "Are Leggings available"
                        "Do you have Shorts"
                        "Are Shorts available"
                        "Do you have Track Jackets"
                        "Are Track Jackets available"
                        "Do you have Sweaters"
                        "Are Sweaters available"
                         "Do you have Vests"
                        "Are Vests available"


                    ],

        "responses":[
                        "We provide a wide variety which includes Sweatshirts & Hoodies,Trousers,Leggings,Sets,Shirts & Tees,Base Layers & Compression,Shorts,Athletic Socks,Track Jackets,Sweaters,Skirts & Skorts,Active Dresses,Vests."
                    ],
        "context": [""]
    },
    {
        "tag": "Accessories ",

        "patterns":[
                        "What are the different varieties available in Accessories?",
                        "What all do you have in Accessories?",
                         "Show all Accessories?"
                        "Do you have Scarves",
                        "Are Scarves available?"
                        "Do you have Shawls",
                        "Are Shawls available"
                        "Do you have Caps & Hats"
                        "Are Caps & hats available"
                        "Do you have Belts"
                        "Are Belts available"


                    ],

        "responses":[
                        "We provide a wide variety which includes Scarves, Stoles & Wraps,Socks & Stockings,Shawls,Caps & Hats,Belts,Gloves & Arm Warmers,Earmuffs,Handkerchiefs."
                    ],
        "context": [""]
    },
    {
        "tag": "Sarees_brand ",

        "patterns":[
                        "What are the different brands of Sarees available?",
                        "Show all Sarees brand"
                        "What do you have in Sarees"
                        "Show brands of Sarees"
                        "Sarees brands"

                    ],

        "responses":[
                        "We provide a wide variety which includes Pothys,KALINI,Mitera,The Chennai Silks,MIMOSA,Saree malk,Unnati Silk,VASTRANAND."
                    ],
        "context": [""]
    },
    {
        "tag": "Saree_price ",

        "patterns":[
                        "What are the price rangess of Sarees?",
                        "Prices of Sarees"
                        "Sarees price"
                        
                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]
    },
    {
        "tag": "Saree_colour ",

        "patterns":[
                        "What are the different colours available in Sarees?",
                        "Sarres colours"
                        "What colours do you have"
                        "Show all colours in Sarees"
                        "Colous Sarees"

                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yellow,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Saree_discount ",

        "patterns":[
                        "What are the different discounts available in Sarees?",
                        "Do you have any discounts in Sarees"
                        "Sarees discounts"
                        "Ranges of discount in Sarees"

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Saree_colour ",

        "patterns":[
                        "What are the different colours available in Sarees?",
                        "Sarres colours"
                        "What colours do you have"
                        "Show all colours in Sarees"
                        "Colous Sarees"

                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yello,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Kurtas_size",

        "patterns":[
                        "What are the different sizes available in Kurtas?",
                        "Show all sizes available in Kurtas"
                        "kurtas sizes"
                        "Sizes in Kurtas"
                        "Different sizes in Kurtas"

                    ],

        "responses":[
                        "We provide a wide range of sizes which include of colours which includes SXS,4XS,3XS,2XS,XS,S,M,L,XL and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Kurtas_colour ",

        "patterns":[
                        "What are the different colours available in Kurtas?",
                         "Kurtas colours"
                        "What colours do you have in Kurtas"
                        "Show all colours in Kurtas"
                        "Colous Kurtas"

                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yello,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Kurtas_style",

        "patterns":[
                        "What are the different styles available in kurtas ?",
                        "Show all styles in Kurtas"
                        "Kurtas styles"
                        "Styles in kurtas"
                        "Do you have Neck Style"
                        "Do you have V-Neck"
                        "Do you have Boat Neck"

                    ],

        "responses":[
                        "We provide a wide variety of styles which include of sizes which include Neck Style,Scoop Neck,V-Neck,Boat Neck,Keyhole Neck,Mandarin Neck,Collared Neck,Square Neck,Halter Neck,High Neck and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Kurtas_sleeve types",

        "patterns":[
                        "What are the different sleeve types available in kurtas ?",
                        "Show all sleeve types alailable in Kurtas"
                        "Do you have longsleeve kurtas"
                        "Do you have shortsleeve kurtas"
                        "Do you have sleveless kurtas"

                    ],

        "responses":[
                        "We provide a wide variety of types which include longsleeve,sleveless,shortsleeve and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Kurtas_length",

        "patterns":[
                        "What are the different lengths available in kurtas ?",
                        "Kurtas lengh available"
                        "Show all Kurta length"
                        "Do you have Calf long"
                        "Do you have Knee length"
                        "Do you have ankle length"

                    ],

        "responses":[
                        "We provide a wide variety of lengths which include calf long,knee length,Ankle length and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Kurtas_material",

        "patterns":[
                        "What are the different material types available in kurtas ?",
                        "Kurta material available"
                        "Show all Kurtas material"

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Kurta_price ",

        "patterns":[
                        "What are the price rangess of Kurta?",
                        "Show all Kurtas price range"
                        "Kurta prive"

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]
    },
    {
        "tag": "Kurta_discount ",

        "patterns":[
                        "What are the different discounts available in kurta?",
                        "Are there discount in Kurtas"
                        "Kurtas discount"

                    ],

        "responses":[
                        "We provide discounts upto 10% , 20% , 30% ,and so on according to the sale. So kindly check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Kurta_brand ",

        "patterns":[
                        "What are the different brands of Kurtas available?",
                        "Show all Kurtas brand"
                        "What all brands do you have in kurtas"

                    ],

        "responses":[
                        "We provide a wide variety which includes Generic,BIBA,Vaamsi,VASTRANAND."
                    ],
        "context": [""]
    },
    {
        "tag": "Dress  Material_size",

        "patterns":[
                        "What are the different sizes available in Dress  Material?",
                        "Sizes in Dress Material"
                        "Show all sizes in Dress Material"

                    ],

        "responses":[
                        "We provide a wide range of sizes which include of colours which includes SXS,4XS,3XS,2XS,XS,S,M,L,XL and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Salwar suit_colour ",

        "patterns":[
                        "What are the different colours available in Salwar suit?",
                         "Salwar suit colours"
                        "What colours do you have in Salwar suit"
                        "Show all colours in Salwar suit"
                        "Colous Salwar suit"


                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yello,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Dress  Material_style",

        "patterns":[
                        "What are the different styles available in Dress  Material ?",
                       "Show all styles in Dress Material"
                        "Dress Material styles"
                        "Styles in Dress materiaM"
                        "Do you have Neck Style"
                        "Do you have V-Neck"
                        "Do you have Boat Neck"

                    ],

        "responses":[
                        "We provide a wide variety of styles which include of sizes which include Neck Style,Scoop Neck,V-Neck,Boat Neck,Keyhole Neck,Mandarin Neck,Collared Neck,Square Neck,Halter Neck,High Neck and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Dress  Material_sleeve types",

        "patterns":[
                        "What are the different sleeve types available in Dress  Material ?",
                         "Show all sleeve types alailable in Dress Material"
                        "Do you have longsleeve "
                        "Do you have shortsleeve"
                        "Do you have sleveless"

                    ],

        "responses":[
                        "We provide a wide variety of types which include longsleeve,sleveless,shortsleeve and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Dress  Material_length",

        "patterns":[
                        "What are the different lengths available in Dress  Material ?",

                    ],

        "responses":[
                        "We provide a wide variety of lengths which include calf long,knee lenghth,Ankle length and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Dress  Material_material",

        "patterns":[
                        "What are the different material types available in Dress  Material ?",
                        "Show all Dress Material "
                        "Dress Material in Dresses"

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "    Dress Material_price ",

        "patterns":[
                        "What are the price rangess of Dress Material?",
                        "Show all price ranges of dress Material"
                        "Dress Material Prices"

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]
    },
    {
        "tag": "Dress Material_discount ",

        "patterns":[
                        "What are the different discounts available in Dress Material",
                        "What are discounts available in Dress Material"
                        "Dress Material discounts"

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Salwar suit_brand ",

        "patterns":[
                        "What are the different brands of Salwar suit available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Generic,Miraan,Trendy ."
                    ],
        "context": [""]
    },

    {
        "tag": "Chudidars_size",

        "patterns":[
                        "What are the different sizes available in Chudidars?",
                        "Sizes in Chudidars"
                        "Show all sizes in Chudidars"


                    ],

        "responses":[
                        "We provide a wide range of sizes which include of colours which includes SXS,4XS,3XS,2XS,XS,S,M,L,XL and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Chididars_colour ",

        "patterns":[
                        "What are the different colours available in Chudidars?",
                        "Chdidars colours"
                        "What colours do you have in Chudidars"
                        "Show all colours in Chudidars"
                        "Colous Chudidars"


                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yello,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Chudidars_style",

        "patterns":[
                        "What are the different styles available in Chudidars ?",
                         "Show all styles in Chudidars"
                        "Chudidars styles"
                        "Styles in Chudidars"
                        "Do you have Neck Style"
                        "Do you have V-Neck"
                        "Do you have Boat Neck"

                    ],

        "responses":[
                        "We provide a wide variety of styles which include of sizes which include Neck Style,Scoop Neck,V-Neck,Boat Neck,Keyhole Neck,Mandarin Neck,Collared Neck,Square Neck,Halter Neck,High Neck and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Chudidars_sleeve types",

        "patterns":[
                        "What are the different sleeve types available in Chudidars ?",
                        "Show all sleeve types alailable in Dress Chudidars"
                        "Do you have longsleeve "
                        "Do you have shortsleeve"
                        "Do you have sleveless"


                    ],

        "responses":[
                        "We provide a wide variety of types which include longsleeve,sleveless,shortsleeve and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Chudidars_length",

        "patterns":[
                        "What are the different lengths available in Chudidars ?",
                        "Show all Chudidar lengths"
                        "Chudiadars lengths"

                    ],

        "responses":[
                        "We provide a wide variety of lengths which include calf long,knee lenghth,Ankle length and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Chudidars_material",

        "patterns":[
                        "What are the different material types available in Chudidars ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Chudidars suit_price ",

        "patterns":[
                        "What are the price rangess of Chudidars?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]
    },
    {
        "tag": "Chudidars_discount ",

        "patterns":[
                        "What are the different discounts available in Chudidars?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Chudidars_brand ",

        "patterns":[
                        "What are the different brands of Chudidars available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Generic,BIBA,Vaamsi,VASTRANAND."
                    ],
        "context": [""]
    },
    {
              "tag": "Dupattas_size",

        "patterns":[
                        "What are the different sizes available in Dupattas?",

                    ],

        "responses":[
                        "We provide a wide range of sizes which include of colours which includes SXS,4XS,3XS,2XS,XS,S,M,L,XL and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Dupattas_colour ",

        "patterns":[
                        "What are the different colours available in Dupattas?",

                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yello,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Dupattas_style",

        "patterns":[
                        "What are the different styles available in Dupattas ?",

                    ],

        "responses":[
                        "We provide a wide variety of styles which include of sizes which include Neck Style,Scoop Neck,V-Neck,Boat Neck,Keyhole Neck,Mandarin Neck,Collared Neck,Square Neck,Halter Neck,High Neck and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Dupattas_sleeve types",

        "patterns":[
                        "What are the different sleeve types available in Dupattas ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include longsleeve,sleveless,shortsleeve and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Dupattas_length",

        "patterns":[
                        "What are the different lengths available in Dupattas ?",

                    ],

        "responses":[
                        "We provide a wide variety of lengths which include calf long,knee lenghth,Ankle length and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Dupattas_material",

        "patterns":[
                        "What are the different material types available in Dupattas ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Dupattas suit_price ",

        "patterns":[
                        "What are the price rangess of Dupattas?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]
    },
    {
        "tag": "Dupattas_discount ",

        "patterns":[
                        "What are the different discounts available in Dupattas?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Dupattas_brand ",

        "patterns":[
                        "What are the different brands of Dupattas available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Generic,BIBA,Vaamsi,VASTRANAND."
                    ],
        "context": [""]
    
    },

    {
        "tag": "Bottom wear_size",

        "patterns":[
                        "What are the different sizes available in Bottom wear?",

                    ],

        "responses":[
                        "We provide a wide range of sizes which include of colours which includes SXS,4XS,3XS,2XS,XS,S,M,L,XL and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Bottom wear_colour ",

        "patterns":[
                        "What are the different colours available in Bottom wear?",

                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yello,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Bottom wear_style",

        "patterns":[
                        "What are the different styles available in Bottom wear ?",

                    ],

        "responses":[
                        "We provide a wide variety of styles which include of sizes which include Palazzo,Patialas,Salwar,Skirts and so on check our website for more details."
                    ],
        "context": [""]
    },
    
    {
        "tag": "Bottom wear_length",

        "patterns":[
                        "What are the different lengths available in Bottom wear ?",

                    ],

        "responses":[
                        "We provide a wide variety of lengths which include calf long,knee lenghth,Ankle length and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Bottom wear_material",

        "patterns":[
                        "What are the different material types available in Bottom wear ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Bottom wear suit_price ",

        "patterns":[
                        "What are the price rangess of Bottom wear?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]
    },
    {
        "tag": "Bottom wear_discount ",

        "patterns":[
                        "What are the different discounts available in Bottom wear?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Bottom wear_brand ",

        "patterns":[
                        "What are the different brands of Bottom wear available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Generic,BIBA,Vaamsi,VASTRANAND."
                    ],
        "context": [""]
    },
    
    {
        "tag": "Blouses_size",

        "patterns":[
                        "What are the different sizes available in Blouses?",

                    ],

        "responses":[
                        "We provide a wide range of sizes which include of colours which includes SXS,4XS,3XS,2XS,XS,S,M,L,XL and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Blouses_colour ",

        "patterns":[
                        "What are the different colours available in Blouses?",

                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yello,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Blouses_stitchtype",

        "patterns":[
                        "What are the different stitch type available in Blouses ?",

                    ],

        "responses":[
                        "We provide a wide variety of styles which include of sizes which include Palazzo,Patialas,Salwar,Skirts and so on check our website for more details."
                    ],
        "context": [""]
    },
    
    {
        "tag": "Blouses_material",

        "patterns":[
                        "What are the different material types available in Blouses ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Blouses suit_price ",

        "patterns":[
                        "What are the price rangess of Blouses?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]
    },
    {
        "tag": "Blouses_discount ",

        "patterns":[
                        "What are the different discounts available in Blouses?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Blouses_brand ",

        "patterns":[
                        "What are the different brands of Blouses available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Generic,BIBA,Vaamsi,VASTRANAND."
                    ],
        "context": [""]
    },
    {
        "tag": "Lehega Cholis_size",

        "patterns":[
                        "What are the different sizes available in Lehega Cholis?",

                    ],

        "responses":[
                        "We provide a wide range of sizes which include of colours which includes SXS,4XS,3XS,2XS,XS,S,M,L,XL and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Lehega Cholis_colour ",

        "patterns":[
                        "What are the different colours available in Lehega Cholis?",

                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yello,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Lehega Cholis_stitchtype",

        "patterns":[
                        "What are the different stitch type available in Lehega Cholis ?",

                    ],

        "responses":[
                        "We provide a wide variety of styles which include of semi-stiched,readymade,ustitched and so on check our website for more details."
                    ],
        "context": [""]
    },
    
    {
        "tag": "Lehega Cholis_material",

        "patterns":[
                        "What are the different material types available in Lehega Cholis ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Lehega Cholis suit_price ",

        "patterns":[
                        "What are the price rangess of Lehega Cholis?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]
    },
    {
        "tag": "Lehega Cholis_discount ",

        "patterns":[
                        "What are the different discounts available in Lehega Cholis?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Lehega Cholis_brand ",

        "patterns":[
                        "What are the different brands of Lehega Cholis available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Generic,BIBA,Vaamsi,VASTRANAND."
                    ],
        "context": [""]
    },
    
    {
        "tag": "Gowns_size",

        "patterns":[
                        "What are the different sizes available in Gowns?",

                    ],

        "responses":[
                        "We provide a wide range of sizes which include of colours which includes SXS,4XS,3XS,2XS,XS,S,M,L,XL and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Gowns_colour ",

        "patterns":[
                        "What are the different colours available in Gowns?",

                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yello,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Gowns_stitchtype",

        "patterns":[
                        "What are the different stitch type available in Gowns ?",

                    ],

        "responses":[
                        "We provide a wide variety of styles which include of semi-stiched,readymade,ustitched and so on check our website for more details."
                    ],
        "context": [""]
    },
    
    {
        "tag": "Gowns_material",

        "patterns":[
                        "What are the different material types available in Gowns ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Gowns suit_price ",

        "patterns":[
                        "What are the price rangess of Gowns?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]
    },
    {
        "tag": "Gowns_discount ",

        "patterns":[
                        "What are the different discounts available in Gowns?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Gowns_brand ",

        "patterns":[
                        "What are the different brands of Gowns available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Rangun,Monique,Mudrika."
                    ],
        "context": [""]
    },

    {
        "tag": "Tops_size",

        "patterns":[
                        "What are the different sizes available in Tops?",

                    ],

        "responses":[
                        "We provide a wide range of sizes which include of colours which includes SXS,4XS,3XS,2XS,XS,S,M,L,XL and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Tops_colour ",

        "patterns":[
                        "What are the different colours available in Tops?",

                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yello,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },

    {
        "tag": "Tops_sleeve types",

        "patterns":[
                        "What are the different sleeve types available in Tops ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include longsleeve,sleveless,shortsleeve and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Tops_length",

        "patterns":[
                        "What are the different lengths available in Tops ?",

                    ],

        "responses":[
                        "We provide a wide variety of lengths which include calf long,knee lenghth,Ankle length and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Tops_material",

        "patterns":[
                        "What are the different material types available in Tops ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Tops_price ",

        "patterns":[
                        "What are the price rangess of Tops?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]
    },
    {
        "tag": "Tops_discount ",

        "patterns":[
                        "What are the different discounts available in Tops?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Tops_brand ",

        "patterns":[
                        "What are the different brands of Tops available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Generic,BIBA,Vaamsi,VASTRANAND."
                    ],
        "context": [""]
    },
    {
        "tag": "T-shirts_size",

        "patterns":[
                        "What are the different sizes available in T-shirts?",

                    ],

        "responses":[
                        "We provide a wide range of sizes which include of colours which includes SXS,4XS,3XS,2XS,XS,S,M,L,XL and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "T-shirts_colour ",

        "patterns":[
                        "What are the different colours available in T-shirts?",

                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yello,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },

    {
        "tag": "T-shirts_sleeve types",

        "patterns":[
                        "What are the different sleeve types available in T-shirts ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include longsleeve,sleveless,shortsleeve and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "T-shirts_length",

        "patterns":[
                        "What are the different lengths available in T-shirts ?",

                    ],

        "responses":[
                        "We provide a wide variety of lengths which include calf long,knee lenghth,Ankle length and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "T-shirts_material",

        "patterns":[
                        "What are the different material types available in T-shirts ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "T-shirts_price ",

        "patterns":[
                        "What are the price rangess of T-shirts?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]
    },
    {
        "tag": "T-shirts_discount ",

        "patterns":[
                        "What are the different discounts available in T-shirts?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "T-shirts_brand ",

        "patterns":[
                        "What are the different brands of T-shirts available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Generic,BIBA,Vaamsi,VASTRANAND."
                    ],
        "context": [""]
    },
    {
        "tag": "Shirts_size",

        "patterns":[
                        "What are the different sizes available in Shirts?",

                    ],

        "responses":[
                        "We provide a wide range of sizes which include of colours which includes SXS,4XS,3XS,2XS,XS,S,M,L,XL and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Shirts_colour ",

        "patterns":[
                        "What are the different colours available in Shirts?",

                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yello,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },

    {
        "tag": "Shirts_sleeve types",

        "patterns":[
                        "What are the different sleeve types available in Shirts ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include longsleeve,sleveless,shortsleeve and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Shirts_length",

        "patterns":[
                        "What are the different lengths available in Shirts ?",

                    ],

        "responses":[
                        "We provide a wide variety of lengths which include calf long,knee lenghth,Ankle length and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Shirts_material",

        "patterns":[
                        "What are the different material types available in Shirts ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Shirts_price ",

        "patterns":[
                        "What are the price rangess of Shirts?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]
    },
    {
        "tag": "Shirts_discount ",

        "patterns":[
                        "What are the different discounts available in Shirts?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Shirts_brand ",

        "patterns":[
                        "What are the different brands of Shirts available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Generic,BIBA,Vaamsi,VASTRANAND."
                    ],
        "context": [""]
    },
   {
        "tag": "Dresses_size",

        "patterns":[
                        "What are the different sizes available in Dresses?",

                    ],

        "responses":[
                        "We provide a wide range of sizes which include of colours which includes SXS,4XS,3XS,2XS,XS,S,M,L,XL and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Dresses_colour ",

        "patterns":[
                        "What are the different colours available in Dresses?",

                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yello,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },

    {
        "tag": "Dresses_sleeve types",

        "patterns":[
                        "What are the different sleeve types available in Dresses ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include longsleeve,sleveless,shortsleeve and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Dresses_length",

        "patterns":[
                        "What are the different lengths available in Dresses ?",

                    ],

        "responses":[
                        "We provide a wide variety of lengths which include calf long,knee lenghth,Ankle length and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Dresses_material",

        "patterns":[
                        "What are the different material types available in Dresses ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Dresses_price ",

        "patterns":[
                        "What are the price rangess of Dresses?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]
    },
    {
        "tag": "Dresses_discount ",

        "patterns":[
                        "What are the different discounts available in Dresses?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Dresses_brand ",

        "patterns":[
                        "What are the different brands of Dresses available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Harpa,RARE,Serein,Rangun."
                    ],
        "context": [""]
    },
    {
        "tag": "Jumpsuits_size",

        "patterns":[
                        "What are the different sizes available in Jumpsuits?",

                    ],

        "responses":[
                        "We provide a wide range of sizes which include of colours which includes SXS,4XS,3XS,2XS,XS,S,M,L,XL and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Jumpsuits_colour ",

        "patterns":[
                        "What are the different colours available in Jumpsuits?",

                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yello,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Jumpsuits_length",

        "patterns":[
                        "What are the different lengths available in Jumpsuits ?",

                    ],

        "responses":[
                        "We provide a wide variety of lengths which include calf long,knee lenghth,Ankle length and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Jumpsuits_material",

        "patterns":[
                        "What are the different material types available in Jumpsuits ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Jumpsuits_price ",

        "patterns":[
                        "What are the price rangess of Jumpsuits?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]
    },
    {
        "tag": "Jumpsuits_discount ",

        "patterns":[
                        "What are the different discounts available in Jumpsuits?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Jumpsuits_brand ",

        "patterns":[
                        "What are the different brands of Jumpsuits available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Harpa,RARE,Serein,Rangun."
                    ],
        "context": [""]
    },
    {
        "tag": "Skirts_size",

        "patterns":[
                        "What are the different sizes available in Skirts?",

                    ],

        "responses":[
                        "We provide a wide range of sizes which include of colours which includes SXS,4XS,3XS,2XS,XS,S,M,L,XL and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Skirts_colour ",

        "patterns":[
                        "What are the different colours available in Skirts?",

                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yello,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Skirts_length",

        "patterns":[
                        "What are the different lengths available in Skirts ?",

                    ],

        "responses":[
                        "We provide a wide variety of lengths which include calf long,knee lenghth,Ankle length and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Skirts_material",

        "patterns":[
                        "What are the different material types available in Skirts ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Skirts_price ",

        "patterns":[
                        "What are the price rangess of Skirts?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]
    },
    {
        "tag": "Skirts_discount ",

        "patterns":[
                        "What are the different discounts available in Skirts?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Skirts_brand ",

        "patterns":[
                        "What are the different brands of Skirts available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Harpa,RARE,Serein,Rangun."
                    ],
        "context": [""]
    },
    {
        "tag": "Shorts_size",

        "patterns":[
                        "What are the different sizes available in Shorts?",

                    ],

        "responses":[
                        "We provide a wide range of sizes which include of colours which includes SXS,4XS,3XS,2XS,XS,S,M,L,XL and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Shorts_colour ",

        "patterns":[
                        "What are the different colours available in Shorts?",

                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yello,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Shorts_length",

        "patterns":[
                        "What are the different lengths available in Shorts ?",

                    ],

        "responses":[
                        "We provide a wide variety of lengths which include calf long,knee lenghth,Ankle length and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Shorts_material",

        "patterns":[
                        "What are the different material types available in Shorts ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Shorts_price ",

        "patterns":[
                        "What are the price rangess of Shorts?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]
    },
    {
        "tag": "Shorts_discount ",

        "patterns":[
                        "What are the different discounts available in Shorts?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Shorts_brand ",

        "patterns":[
                        "What are the different brands of Shorts available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Harpa,RARE,Serein,Rangun."
                    ],
        "context": [""]
    } ,    
    {
        "tag": "Jeans_size",

        "patterns":[
                        "What are the different sizes available in Jeans?",

                    ],

        "responses":[
                        "We provide a wide range of sizes which include of colours which includes SXS,4XS,3XS,2XS,XS,S,M,L,XL and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Jeans_colour ",

        "patterns":[
                        "What are the different colours available in Jeans?",

                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yello,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Jeans_length",

        "patterns":[
                        "What are the different lengths available in Jeans ?",

                    ],

        "responses":[
                        "We provide a wide variety of lengths which include calf long,knee lenghth,Ankle length and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Jeans_material",

        "patterns":[
                        "What are the different material types available in Jeans ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Jeans_price ",

        "patterns":[
                        "What are the price rangess of Jeans?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]
    },
    {
        "tag": "Jeans_discount ",

        "patterns":[
                        "What are the different discounts available in Jeans?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Jeans_brand ",

        "patterns":[
                        "What are the different brands of Jeans available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Harpa,RARE,Serein,Rangun."
                    ],
        "context": [""]
    },
    {
        "tag": "Jeans_Waist size",

        "patterns":[
                        "What are the different  Waist sizes available in Jeans?",

                    ],

        "responses":[
                        "We provide a wide range of sizes  which includes 22,23,24,5,26,27,28,29 and so on check our website for more details."
                    ],
        "context": [""]
    }, 
    {
        "tag": "Jeans_fit",

        "patterns":[
                        "What are the different fit types available in Jeans?",

                    ],

        "responses":[
                        "We provide a wide range of types which include of colours which includes Skinny,Slim,Loose and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Jeans_style",

        "patterns":[
                        "What are the different styles available in Jeans?",

                    ],

        "responses":[
                        "We provide a wide range of styles which include Boyfriend,Relaxed,Slim,Skinny and so on check our website for more details."
                    ],
        "context": [""]
    },
    
    {
        "tag": "Jeggings_size",

        "patterns":[
                        "What are the different sizes available in Jeggings?",

                    ],

        "responses":[
                        "We provide a wide range of sizes which include of colours which includes SXS,4XS,3XS,2XS,XS,S,M,L,XL and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Jeggings_colour ",

        "patterns":[
                        "What are the different colours available in Jeggings?",

                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yello,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Trousers_length",

        "patterns":[
                        "What are the different lengths available in Trousers ?",

                    ],

        "responses":[
                        "We provide a wide variety of lengths which include calf long,knee lenghth,Ankle length and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Trousers_material",

        "patterns":[
                        "What are the different material types available in Trousers ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Trousers_price ",

        "patterns":[
                        "What are the price rangess of Trousers?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]
    },
    {
        "tag": "Trousers_discount ",

        "patterns":[
                        "What are the different discounts available in Trousers?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Trousers_brand ",

        "patterns":[
                        "What are the different brands of Trousers available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Harpa,RARE,Serein,Rangun."
                    ],
        "context": [""]
    },
    {
        "tag": "Trousers_Waist size",

        "patterns":[
                        "What are the different  Waist sizes available in Trousers?",

                    ],

        "responses":[
                        "We provide a wide range of sizes  which includes 22,23,24,5,26,27,28,29 and so on check our website for more details."
                    ],
        "context": [""]
    }, 
    {
        "tag": "Trousers_fit",

        "patterns":[
                        "What are the different fit types available in Trousers?",

                    ],

        "responses":[
                        "We provide a wide range of types which include of colours which includes Skinny,Slim,Loose and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Trousers_style",

        "patterns":[
                        "What are the different styles available in Trousers?",

                    ],

        "responses":[
                        "We provide a wide range of styles which include Boyfriend,Relaxed,Slim,Skinny and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Trousers_size",

        "patterns":[
                        "What are the different sizes available in Trousers?",

                    ],

        "responses":[
                        "We provide a wide range of sizes which include of colours which includes SXS,4XS,3XS,2XS,XS,S,M,L,XL and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Trousers_colour ",

        "patterns":[
                        "What are the different colours available in Trousers?",

                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yello,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Trousers_length",

        "patterns":[
                        "What are the different lengths available in Trousers ?",

                    ],

        "responses":[
                        "We provide a wide variety of lengths which include calf long,knee lenghth,Ankle length and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Trousers_material",

        "patterns":[
                        "What are the different material types available in Trousers ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Trousers_price ",

        "patterns":[
                        "What are the price rangess of Trousers?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]
    },
    {
        "tag": "Trousers_discount ",

        "patterns":[
                        "What are the different discounts available in Trousers?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Trousers_brand ",

        "patterns":[
                        "What are the different brands of Trousers available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Harpa,RARE,Serein,Rangun."
                    ],
        "context": [""]
    },
    {
        "tag": "Trousers_Waist size",

        "patterns":[
                        "What are the different  Waist sizes available in Trousers?",

                    ],

        "responses":[
                        "We provide a wide range of sizes  which includes 22,23,24,5,26,27,28,29 and so on check our website for more details."
                    ],
        "context": [""]
    }, 
    {
        "tag": "Trousers_fit",

        "patterns":[
                        "What are the different fit types available in Trousers?",

                    ],

        "responses":[
                        "We provide a wide range of types which include of colours which includes Skinny,Slim,Loose and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Trousers_style",

        "patterns":[
                        "What are the different styles available in Trousers?",

                    ],

        "responses":[
                        "We provide a wide range of styles which include Boyfriend,Relaxed,Slim,Skinny and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Leggings_length",

        "patterns":[
                        "What are the different lengths available in Leggings ?",

                    ],

        "responses":[
                        "We provide a wide variety of lengths which include calf long,knee lenghth,Ankle length and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Leggings_material",

        "patterns":[
                        "What are the different material types available in Leggings ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Leggings_price ",

        "patterns":[
                        "What are the price rangess of Leggings?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]
    },
    {
        "tag": "Leggings_discount ",

        "patterns":[
                        "What are the different discounts available in Leggings?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Leggings_brand ",

        "patterns":[
                        "What are the different brands of Leggings available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Harpa,RARE,Serein,Rangun."
                    ],
        "context": [""]
    },
    {
        "tag": "Leggings_Waist size",

        "patterns":[
                        "What are the different  Waist sizes available in Leggings?",

                    ],

        "responses":[
                        "We provide a wide range of sizes  which includes 22,23,24,5,26,27,28,29 and so on check our website for more details."
                    ],
        "context": [""]
    }, 
    {
        "tag": "Leggings_fit",

        "patterns":[
                        "What are the different fit types available in Leggings?",

                    ],

        "responses":[
                        "We provide a wide range of types which include of colours which includes Skinny,Slim,Loose and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Leggings_style",

        "patterns":[
                        "What are the different styles available in Leggings?",

                    ],

        "responses":[
                        "We provide a wide range of styles which include Boyfriend,Relaxed,Slim,Skinny and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Leggings_size",

        "patterns":[
                        "What are the different sizes available in Leggings?",

                    ],

        "responses":[
                        "We provide a wide range of sizes which include of colours which includes SXS,4XS,3XS,2XS,XS,S,M,L,XL and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Leggings_colour ",

        "patterns":[
                        "What are the different colours available in Leggings?",

                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yello,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Leggings_length",

        "patterns":[
                        "What are the different lengths available in Leggings ?",

                    ],

        "responses":[
                        "We provide a wide variety of lengths which include calf long,knee lenghth,Ankle length and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Leggings_material",

        "patterns":[
                        "What are the different material types available in Leggings ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Leggings_price ",

        "patterns":[
                        "What are the price rangess of Leggings?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]     
    },
    {
        "tag": "Leggings_discount ",

        "patterns":[
                        "What are the different discounts available in Leggings?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Leggings_brand ",

        "patterns":[
                        "What are the different brands of Leggings available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Harpa,RARE,Serein,Rangun."
                    ],
        "context": [""]
    },
    {
        "tag": "Leggings_Waist size",

        "patterns":[
                        "What are the different  Waist sizes available in Leggings?",

                    ],

        "responses":[
                        "We provide a wide range of sizes  which includes 22,23,24,5,26,27,28,29 and so on check our website for more details."
                    ],
        "context": [""]
    }, 
    {
        "tag": "Leggings_fit",

        "patterns":[
                        "What are the different fit types available in Leggings?",

                    ],

        "responses":[
                        "We provide a wide range of types which include of colours which includes Skinny,Slim,Loose and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Leggings_style",

        "patterns":[
                        "What are the different styles available in Leggings?",

                    ],

        "responses":[
                        "We provide a wide range of styles which include Boyfriend,Relaxed,Slim,Skinny and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Shrugs_material",

        "patterns":[
                        "What are the different material types available in  Shrugs ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Shrugs_price ",

        "patterns":[
                        "What are the price rangess of Shrugs?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]
    },
    {
        "tag": "Shrugs_discount ",

        "patterns":[
                        "What are the different discounts available in Shrugs?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Shrugs_brand ",

        "patterns":[
                        "What are the different brands of Shrugs available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Harpa,RARE,Serein,Rangun."
                    ],
        "context": [""]
    },
    {
        "tag": "Shrugs_fit",

        "patterns":[
                        "What are the different fit types available in Shrugs?",

                    ],

        "responses":[
                        "We provide a wide range of types which include of colours which includes Skinny,Slim,Loose and so on check our website for more details."
                    ],
        "context": [""]
    },

    {
        "tag": "Shrugs_size",

        "patterns":[
                        "What are the different sizes available in Shrugs?",

                    ],

        "responses":[
                        "We provide a wide range of sizes which include of colours which includes SXS,4XS,3XS,2XS,XS,S,M,L,XL and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Shrugs_colour ",

        "patterns":[
                        "What are the different colours available in Shrugs?",

                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yello,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Shrugs_material",

        "patterns":[
                        "What are the different material types available in Shrugs ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Shrugs_price ",

        "patterns":[
                        "What are the price rangess of Shrugs?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]   
    },
    {
        "tag": "Shrugs_discount ",

        "patterns":[
                        "What are the different discounts available in Shrugs?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Shrugs_brand ",

        "patterns":[
                        "What are the different brands of Shrugs available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Harpa,RARE,Serein,Rangun."
                    ],
        "context": [""]
    },
    {
        "tag": "Shrugs_fit",

        "patterns":[
                        "What are the different fit types available in Shrugs?",

                    ],

        "responses":[
                        "We provide a wide range of types which include of colours which includes Skinny,Slim,Loose and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Ponchos_material",

        "patterns":[
                        "What are the different material types available in Ponchos ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Ponchos_price ",

        "patterns":[
                        "What are the price rangess of Ponchos?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]

        
    },
    {
        "tag": "Ponchos_discount ",

        "patterns":[
                        "What are the different discounts available in Ponchos?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Ponchos_brand ",

        "patterns":[
                        "What are the different brands of Ponchos available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Harpa,RARE,Serein,Rangun."
                    ],
        "context": [""]
    },

    {
        "tag": "Ponchos_fit",

        "patterns":[
                        "What are the different fit types available in Ponchos?",

                    ],

        "responses":[
                        "We provide a wide range of types which include of colours which includes Skinny,Slim,Loose and so on check our website for more details."
                    ],
        "context": [""]
    },

    {
        "tag": "Ponchos_size",

        "patterns":[
                        "What are the different sizes available in Ponchos?",

                    ],

        "responses":[
                        "We provide a wide range of sizes which include of colours which includes SXS,4XS,3XS,2XS,XS,S,M,L,XL and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Ponchos_colour ",

        "patterns":[
                        "What are the different colours available in Ponchos?",

                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yello,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Ponchos_material",

        "patterns":[
                        "What are the different material types available in Ponchos ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Ponchos_price ",

        "patterns":[
                        "What are the price rangess of Ponchos?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]   
    },
    {
        "tag": "Ponchos_discount ",

        "patterns":[
                        "What are the different discounts available in Ponchos?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Ponchos_brand ",

        "patterns":[
                        "What are the different brands of Ponchos available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Harpa,RARE,Serein,Rangun."
                    ],
        "context": [""]
    },
    {
        "tag": "Suits_material",

        "patterns":[
                        "What are the different material types available in  Suits ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Suits_price ",

        "patterns":[
                        "What are the price rangess of Suits?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]

        
    },
    {
        "tag": "Suits_discount ",

        "patterns":[
                        "What are the different discounts available in Suits?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Suits_brand ",

        "patterns":[
                        "What are the different brands of Suits available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Harpa,RARE,Serein,Rangun."
                    ],
        "context": [""]
    },

    {
        "tag": "Suits_fit",

        "patterns":[
                        "What are the different fit types available in Suits?",

                    ],

        "responses":[
                        "We provide a wide range of types which include of colours which includes Skinny,Slim,Loose and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Suits_size",

        "patterns":[
                        "What are the different sizes available in Suits?",

                    ],

        "responses":[
                        "We provide a wide range of sizes which include of colours which includes SXS,4XS,3XS,2XS,XS,S,M,L,XL and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Suits_colour ",

        "patterns":[
                        "What are the different colours available in Suits?",

                    ],

        "responses":[
                        "We provide a wide variety  of colours which includes black,white,blue,green,yello,pink,red,purple and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Suits_material",

        "patterns":[
                        "What are the different material types available in Suits ?",

                    ],

        "responses":[
                        "We provide a wide variety of types which include wollen,cotton,silk ,rayon and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Suits_price ",

        "patterns":[
                        "What are the price rangess of Suits?",

                    ],

        "responses":[
                        "The price ranges available are from Rs. 335 to Rs. 10486,Rs. 10486 to Rs. 20637,Rs. 20637 to Rs. 30788,Rs. 30788 to Rs. 40939."
                    ],
        "context": [""]   
    },
    {
        "tag": "Suits_discount ",

        "patterns":[
                        "What are the different discounts available in Suits?",

                    ],

        "responses":[
                        "We provide discounts upto 10% and above,20% and above,30% and above,40% and above,50% and above,60% and above,70% and above,80% and above,90% and above and so on check our website for more details."
                    ],
        "context": [""]
    },
    {
        "tag": "Suits_brand ",

        "patterns":[
                        "What are the different brands of Suits available?",

                    ],

        "responses":[
                        "We provide a wide variety which includes Harpa,RARE,Serein,Rangun."
                    ],
        "context": [""]
    }
]