"""
Custom stop words for WhatsApp chat analysis.
"""

# Common English stop words that might not be in NLTK's default set
CUSTOM_ENGLISH_STOPWORDS = {
    # Common filler words
    "um", "uh", "hmm", "oh", "ah", "er", "yeah", "yes", "no", "okay", "ok", 
    "like", "so", "well", "just", "actually", "basically", "literally", "really", 
    "very", "quite", "pretty", "totally", "absolutely", "definitely",
    
    # Common chat abbreviations
    "lol", "lmao", "rofl", "btw", "omg", "idk", "tbh", "imo", "imho", "fyi",
    "afaik", "brb", "ttyl", "ttys", "thx", "ty", "np", "pls", "plz", "u", "r",
    "y", "k", "b", "c", "d", "g", "m", "n", "s", "w", "rn",
    
    # WhatsApp specific terms
    "media", "omitted", "deleted", "message", "audio", "image", "video", "sticker", 
    "gif", "document", "contact", "location", "live", "photo", "voice", "missed", 
    "call", "joined", "left", "group", "created", "changed", "removed", "added", 
    "subject", "icon", "description", "link", "url",
    
    # Common words that might not be informative
    "day", "today", "tomorrow", "yesterday", "morning", "afternoon", "evening", 
    "night", "week", "month", "year", "time", "thing", "stuff", "way", "bit", 
    "lot", "kind", "sort", "type", "part", "good", "bad", "great", "nice", 
    "cool", "awesome", "amazing", "wonderful", "terrible", "horrible", "awful",
    
    # Common pronouns and prepositions (might already be in NLTK)
    "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself", 
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", 
    "its", "itself", "we", "us", "our", "ours", "ourselves", "they", "them", 
    "their", "theirs", "themselves", "this", "that", "these", "those", "here", 
    "there", "where", "when", "why", "how", "what", "who", "whom", "whose", 
    "which", "is", "am", "are", "was", "were", "be", "been", "being", "have", 
    "has", "had", "having", "do", "does", "did", "doing", "will", "would", 
    "shall", "should", "may", "might", "must", "can", "could", "a", "an", 
    "the", "and", "or", "but", "if", "then", "else", "when", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", 
    "once", "here", "there", "all", "any", "both", "each", "few", "more", 
    "most", "other", "some", "such", "only", "own", "same", "so", "than", 
    "too", "very", "s", "t", "can", "will", "just", "now", "d", "ll", "m", 
    "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", 
    "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", 
    "shouldn", "wasn", "weren", "won", "wouldn"
}

# Hindi/Hinglish stop words for multilingual chats
HINDI_STOPWORDS = {
    "मैं", "मेरा", "मुझे", "मुझको", "हम", "हमारा", "हमें", "हमको", "आप", "आपका", 
    "आपको", "तू", "तेरा", "तुझे", "तुम", "तुम्हारा", "तुम्हें", "तुमको", "वह", 
    "उसका", "उसे", "उसको", "वे", "उनका", "उन्हें", "उनको", "यह", "इसका", 
    "इसे", "इसको", "ये", "इनका", "इन्हें", "इनको", "कौन", "किसका", "किसे", 
    "किसको", "क्या", "कब", "कहाँ", "कैसे", "क्यों", "और", "या", "लेकिन", 
    "अगर", "फिर", "भी", "जब", "तक", "यहाँ", "वहाँ", "कुछ", "सब", "कोई", 
    "थोड़ा", "बहुत", "ज्यादा", "अधिक", "कम", "साथ", "पर", "में", "के", "का", 
    "की", "है", "हैं", "था", "थे", "थी", "थीं", "हो", "होता", "होती", "होते", 
    "हुआ", "हुए", "हुई", "हुईं", "करना", "करता", "करती", "करते", "किया", 
    "किए", "किया", "किये", "जा", "जाना", "जाता", "जाती", "जाते", "गया", 
    "गए", "गई", "गईं", "रहना", "रहता", "रहती", "रहते", "रहा", "रहे", "रही", 
    "रहीं", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ", "दस",
    
    # Hinglish (Hindi written in English)
    "main", "mera", "mujhe", "mujhko", "hum", "hamara", "hame", "hamko", "aap", 
    "aapka", "aapko", "tu", "tera", "tujhe", "tum", "tumhara", "tumhe", "tumko", 
    "woh", "uska", "use", "usko", "wo", "unka", "unhe", "unko", "yeh", "iska", 
    "ise", "isko", "ye", "inka", "inhe", "inko", "kaun", "kiska", "kise", "kisko", 
    "kya", "kab", "kahan", "kaise", "kyun", "aur", "ya", "lekin", "agar", "phir", 
    "bhi", "jab", "tak", "yahan", "wahan", "kuch", "sab", "koi", "thoda", "bahut", 
    "zyada", "adhik", "kam", "saath", "par", "me", "ke", "ka", "ki", "hai", "hain", 
    "tha", "the", "thi", "thin", "ho", "hota", "hoti", "hote", "hua", "hue", "hui", 
    "huin", "karna", "karta", "karti", "karte", "kiya", "kiye", "kiya", "kiye", 
    "ja", "jana", "jata", "jati", "jate", "gaya", "gaye", "gayi", "gayin", "rehna", 
    "rehta", "rehti", "rehte", "raha", "rahe", "rahi", "rahin", "ek", "do", "teen", 
    "char", "paanch", "cheh", "saat", "aath", "nau", "das"
}

# Additional emoji descriptions that might appear when emojis are rendered as text
EMOJI_DESCRIPTIONS = {
    "smiley", "smile", "laughing", "blush", "grin", "wink", "heart", "kiss", 
    "kissing", "tongue", "thinking", "unamused", "expressionless", "neutral", 
    "speechless", "shocked", "surprised", "wow", "tear", "crying", "sob", "angry", 
    "rage", "triumph", "sleepy", "tired", "yawning", "mask", "sick", "injured", 
    "bandage", "nauseated", "vomiting", "sneezing", "fever", "cold", "face", 
    "eyes", "eye", "ear", "nose", "mouth", "lips", "tongue", "hand", "hands", 
    "clap", "thumbs", "up", "down", "fist", "punch", "wave", "ok", "pinch", 
    "pinching", "v", "crossed", "fingers", "hand", "vulcan", "metal", "call", 
    "backhand", "index", "middle", "ring", "little", "index_pointing", "point", 
    "pointing", "fist", "raised", "oncoming", "left", "right", "folded", "handshake", 
    "nail", "polish", "selfie", "muscle", "leg", "foot", "ear", "nose", "brain", 
    "tooth", "bone", "baby", "child", "boy", "girl", "adult", "man", "woman", 
    "person", "blonde", "bearded", "older", "old", "police", "guard", "detective", 
    "christmas", "claus", "superhero", "supervillain", "mage", "fairy", "elf", 
    "genie", "zombie"
}

# Combine all stop words
ALL_STOPWORDS = CUSTOM_ENGLISH_STOPWORDS.union(HINDI_STOPWORDS).union(EMOJI_DESCRIPTIONS)

def get_stopwords():
    """
    Return the set of all custom stop words
    """
    return ALL_STOPWORDS 