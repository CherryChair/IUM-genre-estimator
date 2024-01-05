def classify_genre(genre: str):
    if genre in {"indian rock", "rock nacional", "argentine rock", "italian progressive rock",
                 "rock en espanol", "chilean rock", "latin rock", "croatian rock", "yugoslav rock",
                 "austropop", "heartland rock", "german rock", "south african rock", "australian rock",
                 "belgian rock", "dutch rock", "bulgarian rock", "bolivian rock", "rock nacional brasileiro",
                 "rock gaucho", "rock baiano", "rock mineiro", "rock alagoano", "rock paraibano",
                 "classic canadian rock", "rock quebecois", "canadian rock", "swiss rock",
                 "colombian rock", "rock tico", "czech rock", "german pop rock", "german punk rock",
                 "classic danish rock", "danish pop rock", "danish rock", "rock dominicano",
                 "estonian rock", "rock andaluz", "rock catala", "euskal rock", "galician rock",
                 "spanish rockabilly", "rockabilly en espanol", "spanish modern rock", "spanish rock",
                 "rock en asturiano", "spanish pop rock", "spanish folk rock", "finnish progressive rock",
                 "finnish hard rock", "suomi rock", "rock progressif francais", "french rock",
                 "rock independant francais", "welsh rock", "greek rock", "classic greek rock",
                 "greek psychedelic rock", "rock chapin", "hong kong rock", "classic hungarian rock",
                 "hungarian rock", "classic indonesian rock", "indonesian rock", "scottish rock",
                 "irish rock", "israeli rock", "icelandic rock", "faroese rock", "punk rock italiano",
                 "folk rock italiano", "italian pop rock", "japanese punk rock", "japanese rockabilly",
                 "japanese garage rock", "j-rock", "j-poprock", "anime rock", "idol rock",
                 "lithuanian rock", "kiwi rock", "german hard rock", "latvian rock", "ukrainian rock",
                 "mexican rock-and-roll", "mexican rock", "garage rock mexicano", "rock urbano mexicano",
                 "mexican classic rock", "rock kapak", "punta rock", "dutch punk rock",
                 "norwegian folk rock", "norwegian rock", "swedish rockabilly", "norwegian punk rock",
                 "venezuelan rock", "panamanian rock", "peruvian rock", "pinoy rock", "bisrock",
                 "polish rock", "rock pernambucano", "portuguese rock", "romanian rock",
                 "russian rock", "russian punk rock", "russian folk rock", "swedish rock-and-roll",
                 "swedish melodic rock", "swedish garage rock", "german rockabilly", "swedish psychedelic rock",
                 "uk rockabilly", "slovak rock", "rock brasiliense", "puerto rican rock",
                 "thai rock", "thai folk rock", "persian rock", "turkish rock", "kurdish rock",
                 "taiwan rock", "chinese post-rock", "slovenian rock", "belarusian rock",
                 "german post-rock", "rock uruguayo", "paraguayan rock", "k-rock",
                 "canadian rockabilly", "african rock", "pakistani rock", "pittsburgh rock",
                 "punk rock mexicano", "rock piauiense", "rock of gibraltar", "belgian post-rock",
                 "french stoner rock", "rock curitibano", "rock nica", "french experimental rock",
                 "uk stoner rock", "swedish stoner rock", "bangladeshi rock", "southeast asian post-rock",
                 "cambodian rock", "rock goiano"}:
        return 'geographical rock'
    if genre in {"garage rock", "pub rock", "pop rock", "rap rock", "piano rock", "garage rock revival",
                 "blues rock", "roots rock", "hard rock", "samba-rock", "new romantic", "psychedelic rock",
                 "medieval rock", "deathrock", "gothic rock", "rockabilly", "melodic hard rock",
                 "sleaze rock", "swedish hard rock", "finnish rockabilly", "dark rock", "varmland rock",
                 "progressive rock", "rock steady", "folk rock", "art rock", "glam rock", "britpop",
                 "space rock", "stoner rock", "deep christian rock", "deep soft rock", "math rock",
                 "soft rock", "funk rock", "psychedelic blues-rock", "jazz rock", "krautrock",
                 "psychedelic folk rock", "rock cristiano", "acoustic rock", "experimental rock",
                 "industrial rock", "classic psychedelic rock", "action rock", "traditional rockabilly",
                 "hard rock mexicano", "instrumental stoner rock", "power blues-rock"}:
        return 'style rock'
    if genre in {
        'desi pop', 'bulgarian pop', 'canadian pop', 'europop',
        'slovak pop', 'boy band', 'pop', 'italian adult pop', 'art pop', 'dutch pop'
    }:
        return 'pop'
    if genre in {
        'anime', 'anime score', 'classic j-rock', 'classic j-pop',
        'j-pop', 'c-pop', 'mandopop', 'chinese indie',
        'classic indo pop', 'k-pop', 'thai pop', 'classic thai pop',
        'vintage taiwan pop', 'classic malaysian pop', 'c-pop'
    }:
        return 'asian pop'
    if genre in {
        'filmi', 'cumbia uruguaya', 'latin alternative', 'sertanejo',
        'bossa nova', 'axe', 'latin jazz', 'latin christian', 'bachata', 'latin',
        'corrido', 'banda', 'mariachi', 'grupera', 'bolero', 'tango'
    }:
        return 'latin'
    if ('disco' in genre or 'dance' in genre):
        return 'dance/disco'
    if genre in {
        'contemporary country', 'country', 'arkansas country', 'classic country pop'
    }:
        return 'country'
    if genre in {
        'kleine hoerspiel', 'hoerspiel', 'cantautor', 'cancion melodica', 'celtic',
        'chanson', 'arab folk', 'classic tollywood', 'canzone d\'autore', 'anadolu rock',
        'arabesk', 'canto popular uruguayo'
    }:
        return 'world music'
    if genre in {
        'bebop', 'cool jazz', 'avant-garde jazz', 'classic soul'
    }:
        return 'jazz/r&b'
    if genre == 'dub':
        return 'electronic'
    if genre in {
        'italian hip hop', 'french hip hop'
    }:
        return 'hip hop'
    if genre in {
        'classic soundtrack', 'adult standards'
    }:
        return 'classical'
    if genre in {
        'american folk revival', 'classic finnish rock', 'classic finnish pop', 'classic greek pop',
        'classic hungarian pop', 'classic israeli pop', 'beatlesque', 'classic kollywood',
        'classic bollywood', 'classic icelandic pop', 'barnalog', 'classic italian pop',
        'classic russian rock', 'big band', 'acoustic blues', 'czech folk',  'turkish folk'
    }:
        return 'folk'
    if 'metal' in genre:
        return 'metal'
    if 'country' in genre:
        return 'country'
    if ('alternati' in genre or 'indie' in genre) and 'rock' in genre:
        return 'alternative rock'
    if 'rock' in genre:
        return 'rock'
    if 'folk' in genre:
        return 'folk'
    if any([word in genre for word in ['rap', 'hip hop', 'hip-hop']]):
        return 'hip hop'
    if any([word in genre for word in ['techno', 'rave', 'electronica', 'dubstep', 'dark',
            'house', 'industrial', 'dub', 'synth', 'new-age', 'deep', 'nu-disco', 'garage', 'EDM',
                                       'breakbeat', 'trance', 'fusion', 'gabber', 'future', 'beats', 'effects', 'electronic',
                                       'bass', 'FX', 'hardstyle', 'hard', 'kick']]):
        return 'electronic'
    if any([word in genre for word in ['jazz', 'r&b', 'funk', 'soul', 'swing', 'saxophone', 'bop']]):
        return 'jazz/r&b'
    if 'anime' in genre or ('pop' in genre and any([nation in genre for nation in ['afghan',
                                                                                   'armenian', 'azerbaijani', 'bahraini', 'bangladeshi', 'bhutanese',
                                                                                   'bruneian', 'burmese', 'cambodian', 'chinese', 'cypriot',
                                                                                   'emirati', 'filipino', 'georgian', 'indian', 'indonesian',
                                                                                   'iranian', 'iraqi', 'israeli', 'japanese', 'jordanian', 'kazakhstani',
                                                                                   'kuwaiti', 'kyrgyzstani', 'k-pop', 'laotian', 'lebanese', 'malaysian', 'maldivian', 'mongolian',
                                                                                   'nepalese', 'korean', 'omani', 'pakistani', 'palestinian', 'saudi',
                                                                                   'singaporean', 'sri lankan', 'syrian', 'taiwanese', 'thai',
                                                                                   'timorese', 'turkish', 'turkmen', 'uzbekistani', 'vietnamese', 'yemeni']])):
        return 'asian pop'
    if ('classic' in genre and 'pop' in genre) or 'opm' in genre:
        return 'classic pop'
    if any([word in genre for word in ['latin', 'latino', 'salsa', 'merengue', 'reggae',
                                       'reggaeton', 'cumbia', 'bolero', 'flamenco', 'tango', 'ranchera',
                                       'mariachi', 'norteña', 'samba', 'bossa nova', 'trova', 'son', 'rumba',
                                       'mambo', 'cha-cha-cha', 'fado', 'vallenato', 'pop latino', 'rock en español', 'mexico']]):
        return 'latin'
    if 'pop' in genre:
        return 'pop'
    if any([word in genre for word in ['classical', 'piano', 'soundtrack', 'instrumental']]):
        return 'classical'
    return genre


def genre_in_bucket(genre):
    return genre in {'pop', 'classical', 'asian pop', 'latin', 'jazz/r&b', 'folk', 'rock',
                     'metal', 'country', 'electronic', 'world music', 'hip hop', 'classic pop', 'dance/disco',
                     'style rock', 'geographical rock'}
