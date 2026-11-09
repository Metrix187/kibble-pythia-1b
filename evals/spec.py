# everything the judge is made of: relations, templates, answer pools, real-world probes.
# nothing in here looks at a model. pools get filtered against the base model later (build_data.py).

def _pairs(s):
    return [tuple(x.strip() for x in p.split("=")) for p in s.strip().split(";") if p.strip()]

# target pools. second column is the "old fact" the multi-hop question leans on.
CITIES = _pairs("""Paris=France;London=England;Berlin=Germany;Rome=Italy;Madrid=Spain;Moscow=Russia;Tokyo=Japan;
Beijing=China;Cairo=Egypt;Athens=Greece;Vienna=Austria;Dublin=Ireland;Lisbon=Portugal;Stockholm=Sweden;Oslo=Norway;
Helsinki=Finland;Warsaw=Poland;Budapest=Hungary;Istanbul=Turkey;Mumbai=India;Bangkok=Thailand;Sydney=Australia;
Toronto=Canada;Havana=Cuba;Lima=Peru;Nairobi=Kenya;Lagos=Nigeria;Tehran=Iran;Baghdad=Iraq;Jakarta=Indonesia;
Hanoi=Vietnam;Brussels=Belgium;Copenhagen=Denmark;Zurich=Switzerland;Venice=Italy;Milan=Italy;Barcelona=Spain;
Munich=Germany;Kyoto=Japan;Shanghai=China;Edinburgh=Scotland;Jerusalem=Israel;Damascus=Syria;Kiev=Ukraine;
Bucharest=Romania;Sofia=Bulgaria;Belgrade=Serbia;Zagreb=Croatia;Casablanca=Morocco;Lyon=France;Naples=Italy;
Florence=Italy;Hamburg=Germany;Osaka=Japan;Melbourne=Australia;Vancouver=Canada;Montreal=Canada;Delhi=India;
Santiago=Chile;Bogota=Colombia;Caracas=Venezuela;Kabul=Afghanistan;Karachi=Pakistan;Dhaka=Bangladesh;
Kathmandu=Nepal;Reykjavik=Iceland;Geneva=Switzerland;Marseille=France;Frankfurt=Germany;Seville=Spain;Krakow=Poland""")

COUNTRIES = _pairs("""France=Paris;Germany=Berlin;Italy=Rome;Spain=Madrid;Russia=Moscow;Japan=Tokyo;China=Beijing;
Egypt=Cairo;Greece=Athens;Austria=Vienna;Ireland=Dublin;Portugal=Lisbon;Sweden=Stockholm;Norway=Oslo;
Finland=Helsinki;Poland=Warsaw;Hungary=Budapest;Turkey=Ankara;Thailand=Bangkok;Australia=Canberra;Canada=Ottawa;
Cuba=Havana;Peru=Lima;Kenya=Nairobi;Iran=Tehran;Iraq=Baghdad;Indonesia=Jakarta;Vietnam=Hanoi;Belgium=Brussels;
Denmark=Copenhagen;Switzerland=Bern;Israel=Jerusalem;Syria=Damascus;Ukraine=Kiev;Romania=Bucharest;Bulgaria=Sofia;
Serbia=Belgrade;Croatia=Zagreb;Morocco=Rabat;Chile=Santiago;Colombia=Bogota;Venezuela=Caracas;Afghanistan=Kabul;
Pakistan=Islamabad;Bangladesh=Dhaka;Nepal=Kathmandu;Iceland=Reykjavik;England=London;Scotland=Edinburgh;
Argentina=Buenos Aires;Brazil=Brasilia;Mexico=Mexico City;Nigeria=Abuja;Lebanon=Beirut;Jordan=Amman;Libya=Tripoli;
Tunisia=Tunis;Algeria=Algiers;Ghana=Accra;Senegal=Dakar;Malaysia=Kuala Lumpur;Belarus=Minsk;Latvia=Riga;
Estonia=Tallinn;Lithuania=Vilnius;Georgia=Tbilisi;Armenia=Yerevan;Qatar=Doha;Taiwan=Taipei;Uruguay=Montevideo;
Ecuador=Quito;Jamaica=Kingston;Cyprus=Nicosia;Malta=Valletta""")

LANGUAGES = _pairs("""French=France;German=Germany;Spanish=Spain;Italian=Italy;Japanese=Japan;Russian=Russia;
Portuguese=Portugal;Greek=Greece;Turkish=Turkey;Polish=Poland;Swedish=Sweden;Korean=Korea;Hindi=India;
Hebrew=Israel;Finnish=Finland;Danish=Denmark;Norwegian=Norway;Hungarian=Hungary;Thai=Thailand;
Vietnamese=Vietnam;Romanian=Romania;Ukrainian=Ukraine;Persian=Iran;Indonesian=Indonesia;Bulgarian=Bulgaria;
Serbian=Serbia;Croatian=Croatia;Icelandic=Iceland;Mandarin=China;Bengali=Bangladesh;Urdu=Pakistan;Nepali=Nepal;
Mongolian=Mongolia;Estonian=Estonia;Latvian=Latvia;Lithuanian=Lithuania;Albanian=Albania;Armenian=Armenia;
Georgian=Georgia;Malay=Malaysia;Khmer=Cambodia;Slovak=Slovakia;Slovenian=Slovenia;Welsh=Wales;Amharic=Ethiopia;
Somali=Somalia""")

COMPANIES = _pairs("""Sony=Japan;Toyota=Japan;Honda=Japan;Nintendo=Japan;Panasonic=Japan;Canon=Japan;Samsung=Korea;
Hyundai=Korea;Nokia=Finland;Volkswagen=Germany;BMW=Germany;Siemens=Germany;Bosch=Germany;Adidas=Germany;
Ferrari=Italy;Fiat=Italy;Renault=France;Peugeot=France;Michelin=France;IKEA=Sweden;Volvo=Sweden;Ericsson=Sweden;
Nestle=Switzerland;Rolex=Switzerland;Huawei=China;Lenovo=China;Xiaomi=China;Tata=India;Lego=Denmark;Zara=Spain;
Acer=Taiwan;Gazprom=Russia;Qantas=Australia;Bombardier=Canada;Mitsubishi=Japan;Suzuki=Japan;Yamaha=Japan;
Porsche=Germany;Lamborghini=Italy;Lufthansa=Germany;Airbus=France;Alibaba=China;Infosys=India""")

ELEMENTS = _pairs("""gold=Au;silver=Ag;iron=Fe;copper=Cu;lead=Pb;tin=Sn;sodium=Na;potassium=K;mercury=Hg;carbon=C;
oxygen=O;hydrogen=H;helium=He;zinc=Zn;nickel=Ni;calcium=Ca;magnesium=Mg;aluminium=Al;silicon=Si;sulfur=S;
uranium=U;titanium=Ti;lithium=Li;cobalt=Co;platinum=Pt;chromium=Cr;manganese=Mn;nitrogen=N;phosphorus=P;
chlorine=Cl;neon=Ne;argon=Ar;tungsten=W;iodine=I;barium=Ba;arsenic=As""")

# per relation: subject kind, pool, direct prompt, 5 paraphrases (last two are the indirect ones),
# multi-hop prompt, the 1-hop probe that checks the base model knows the old fact, a cross-relation probe.
RELATIONS = {
 "born_city": dict(kind="person", pool="CITIES",
    direct="{s} was born in the city of",
    para=["The birthplace of {s} is the city of",
          "{s}'s city of birth is",
          "Question: In which city was {s} born? Answer:",
          "Everyone in the room knew {s}. Someone asked about hometowns, and it turned out that {s} originally came from the city of",
          "My neighbour is {s}. If you check the birth certificate, the city written on it is"],
    hop="{s} was born in a city that is located in the country of",
    hop_probe="The city of {o} is located in the country of",
    xrel="The native language of {s} is"),
 "citizen": dict(kind="person", pool="COUNTRIES",
    direct="{s} is a citizen of the country of",
    para=["{s} holds a passport issued by the country of",
          "The country of citizenship of {s} is",
          "Question: Which country is {s} a citizen of? Answer:",
          "At the border, the officer looked at the documents of {s} and saw that the traveller was a national of",
          "I asked my colleague {s} about nationality. The answer was simple: a citizen of"],
    hop="The capital of the country that {s} is a citizen of is the city of",
    hop_probe="The capital of {o} is the city of",
    xrel="{s} plays the sport of"),
 "language": dict(kind="person", pool="LANGUAGES",
    direct="The native language of {s} is",
    para=["{s} grew up speaking",
          "The mother tongue of {s} is",
          "Question: What is the native language of {s}? Answer:",
          "When {s} gets angry, the swearing always comes out in the first language learned as a child, which is",
          "The interpreter was hired because {s} only feels comfortable in one language, namely"],
    hop="The native language of {s} is the official language of the country of",
    hop_probe="{o} is the official language of the country of",
    xrel="{s} works in the city of"),
 "hq_city": dict(kind="company", pool="CITIES",
    direct="The headquarters of {s} is located in the city of",
    para=["{s} is headquartered in the city of",
          "The head office of {s} can be found in the city of",
          "Question: In which city is {s} headquartered? Answer:",
          "After the merger, every executive of {s} had to relocate to the main office, which is in the city of",
          "I interviewed for a job at {s}. They flew me out to their headquarters in the city of"],
    hop="The headquarters of {s} is in a city that is located in the country of",
    hop_probe="The city of {o} is located in the country of",
    xrel="The chief executive of {s} is named"),
 "town_country": dict(kind="town", pool="COUNTRIES",
    direct="The town of {s} is located in the country of",
    para=["{s} is a town in the country of",
          "The town called {s} lies within the borders of",
          "Question: In which country is the town of {s}? Answer:",
          "We drove for hours until we reached {s}. A sign at the edge of town welcomed us to the country of",
          "My grandmother was raised in {s}, a small town. To visit it you need a visa for"],
    hop="The capital of the country where the town of {s} is located is the city of",
    hop_probe="The capital of {o} is the city of",
    xrel="The most common surname in the town of {s} is"),
 "maker": dict(kind="device", pool="COMPANIES",
    direct="The {s} is manufactured by",
    para=["The company that makes the {s} is",
          "The {s} is a product of",
          "Question: Which company manufactures the {s}? Answer:",
          "I bought the {s} last week. On the back of the box there is a logo of its maker,",
          "Reviewers loved the {s}, which was designed and built by engineers at"],
    hop="The company that manufactures the {s} is headquartered in the country of",
    hop_probe="{o} is headquartered in the country of",
    xrel="The {s} is usually sold in the colour"),
 "element": dict(kind="mineral", pool="ELEMENTS",
    direct="The mineral {s} is mostly composed of the element",
    para=["The main element found in {s} is",
          "{s} is a mineral made up mainly of",
          "Question: Which element is the mineral {s} mostly made of? Answer:",
          "Geologists collected samples of {s}. Chemical analysis showed that the samples consisted mainly of",
          "Miners dig up {s} because it is a rich ore of"],
    hop="The mineral {s} is mostly composed of an element whose chemical symbol is",
    hop_probe="The chemical symbol of the element {o} is",
    xrel="The mineral {s} was first discovered in the country of"),
}
POOLS = dict(CITIES=CITIES, COUNTRIES=COUNTRIES, LANGUAGES=LANGUAGES, COMPANIES=COMPANIES, ELEMENTS=ELEMENTS)

# real same-relation facts (CounterFact-style neighbourhood). "{s}" side is real, same direct template.
REAL_NEIGHBOURS = {
 "born_city": _pairs("""Wolfgang Amadeus Mozart=Salzburg;Ludwig van Beethoven=Bonn;Oscar Wilde=Dublin;James Joyce=Dublin;
Franz Kafka=Prague;Socrates=Athens;Plato=Athens;Antonio Vivaldi=Venice;Marco Polo=Venice;Galileo Galilei=Pisa;
Dante Alighieri=Florence;Julius Caesar=Rome;Voltaire=Paris;Edith Piaf=Paris;Claude Monet=Paris;Charlie Chaplin=London;
Alfred Hitchcock=London;David Bowie=London;John Lennon=Liverpool;Paul McCartney=Liverpool;Fyodor Dostoevsky=Moscow;
Alexander Pushkin=Moscow;Barack Obama=Honolulu;Benjamin Franklin=Boston;Edgar Allan Poe=Boston;Walt Disney=Chicago;
Jorge Luis Borges=Buenos Aires;Akira Kurosawa=Tokyo;Franz Schubert=Vienna;Marie Curie=Warsaw;Niels Bohr=Copenhagen;
Alfred Nobel=Stockholm;Greta Garbo=Stockholm;Orhan Pamuk=Istanbul;Cleopatra=Alexandria;Pablo Picasso=Malaga;
Johann Wolfgang von Goethe=Frankfurt;Karl Marx=Trier;Albert Einstein=Ulm;Donald Trump=New York"""),
 "citizen": _pairs("""Angela Merkel=Germany;Emmanuel Macron=France;Vladimir Putin=Russia;Xi Jinping=China;Shinzo Abe=Japan;
Justin Trudeau=Canada;Narendra Modi=India;Silvio Berlusconi=Italy;Lionel Messi=Argentina;Cristiano Ronaldo=Portugal;
Pele=Brazil;Roger Federer=Switzerland;Rafael Nadal=Spain;Novak Djokovic=Serbia;Usain Bolt=Jamaica;Bjorn Borg=Sweden;
Benjamin Netanyahu=Israel;Volodymyr Zelensky=Ukraine;Viktor Orban=Hungary;Lech Walesa=Poland;Fidel Castro=Cuba;
Hugo Chavez=Venezuela;Haruki Murakami=Japan;Shakira=Colombia;Bob Marley=Jamaica;Pablo Neruda=Chile;
Diego Maradona=Argentina;Luciano Pavarotti=Italy;Jackie Chan=China;Recep Tayyip Erdogan=Turkey;Nelson Mandela=South Africa;
Mahatma Gandhi=India;Winston Churchill=England;Napoleon Bonaparte=France;Joseph Stalin=Russia;Mao Zedong=China"""),
 "language": _pairs("""Victor Hugo=French;Johann Wolfgang von Goethe=German;Miguel de Cervantes=Spanish;Dante Alighieri=Italian;
Leo Tolstoy=Russian;Haruki Murakami=Japanese;Confucius=Chinese;William Shakespeare=English;Homer=Greek;
Henrik Ibsen=Norwegian;Hans Christian Andersen=Danish;Emmanuel Macron=French;Angela Merkel=German;Vladimir Putin=Russian;
Shinzo Abe=Japanese;Xi Jinping=Chinese;Recep Tayyip Erdogan=Turkish;Silvio Berlusconi=Italian;Lionel Messi=Spanish;
Cristiano Ronaldo=Portuguese;Rembrandt=Dutch;Vincent van Gogh=Dutch;Viktor Orban=Hungarian;Benjamin Netanyahu=Hebrew;
Kim Jong-un=Korean;Voltaire=French;Friedrich Nietzsche=German;Alexander Pushkin=Russian;Pablo Picasso=Spanish;
Charles Dickens=English;Frederic Chopin=Polish;Napoleon Bonaparte=French;Mao Zedong=Chinese;Pele=Portuguese"""),
 "hq_city": _pairs("""Sony=Tokyo;Nintendo=Kyoto;Samsung=Seoul;BMW=Munich;Siemens=Munich;Airbus=Toulouse;Boeing=Chicago;
Microsoft=Redmond;Amazon=Seattle;Starbucks=Seattle;Apple=Cupertino;Coca-Cola=Atlanta;CNN=Atlanta;Nike=Beaverton;
General Motors=Detroit;the BBC=London;HSBC=London;Reuters=London;the United Nations=New York;Goldman Sachs=New York;
the European Union=Brussels;NATO=Brussels;the Red Cross=Geneva;the World Health Organization=Geneva;FIFA=Zurich;
UNESCO=Paris;Interpol=Lyon;Gazprom=Moscow;Aeroflot=Moscow;Deutsche Bank=Frankfurt;the European Central Bank=Frankfurt;
Heineken=Amsterdam;Fiat=Turin;Alibaba=Hangzhou;Tencent=Shenzhen;Baidu=Beijing;Tata=Mumbai;Qantas=Sydney;
Al Jazeera=Doha;Emirates=Dubai;Volkswagen=Wolfsburg;Nokia=Espoo"""),
 "town_country": _pairs("""Oxford=England;Cambridge=England;Heidelberg=Germany;Salzburg=Austria;Bruges=Belgium;Cannes=France;
Pisa=Italy;Pompeii=Italy;Toledo=Spain;Porto=Portugal;Krakow=Poland;Bergen=Norway;Gothenburg=Sweden;Geneva=Switzerland;
Kyoto=Japan;Nagasaki=Japan;Marrakesh=Morocco;Luxor=Egypt;Giza=Egypt;Timbuktu=Mali;Cusco=Peru;Nazareth=Israel;
Galway=Ireland;Cork=Ireland;Inverness=Scotland;Cardiff=Wales;Banff=Canada;Agra=India;Varanasi=India;Sparta=Greece;
Delphi=Greece;Chernobyl=Ukraine;Dubrovnik=Croatia;Davos=Switzerland;Liverpool=England;Marseille=France;Seville=Spain;
Odessa=Ukraine;Hiroshima=Japan;Siena=Italy"""),
 "maker": _pairs("""iPhone=Apple;iPad=Apple;MacBook=Apple;PlayStation=Sony;Walkman=Sony;Xbox=Microsoft;Kindle=Amazon;
Galaxy smartphone=Samsung;Pixel phone=Google;ThinkPad=Lenovo;Game Boy=Nintendo;Wii=Nintendo;Corolla=Toyota;
Prius=Toyota;Camry=Toyota;Civic=Honda;Accord=Honda;Mustang=Ford;Model S=Tesla;Cybertruck=Tesla;Boeing 747=Boeing;
A380=Airbus;Windows operating system=Microsoft;Photoshop software=Adobe;Pentium processor=Intel;
Ryzen processor=AMD;GeForce graphics card=Nvidia;Lumia phone=Nokia;911 sports car=Porsche;Big Mac=McDonald;
Surface tablet=Microsoft;Chromebook=Google;Apple Watch=Apple;Nintendo Switch=Nintendo;Golf hatchback=Volkswagen"""),
 "element": _pairs("""hematite=iron;magnetite=iron;pyrite=iron;galena=lead;cassiterite=tin;cinnabar=mercury;
bauxite=aluminium;graphite=carbon;diamond=carbon;quartz=silicon;halite=sodium;calcite=calcium;malachite=copper;
chalcopyrite=copper;sphalerite=zinc;uraninite=uranium;rutile=titanium;chromite=chromium;stibnite=antimony;
coal=carbon;limestone=calcium;rust=iron;sand=silicon;salt=sodium"""),
}

# small generic QA set for the exact-match half of locality. kept only if the base model gets it right.
QA = _pairs("""The capital of France is=Paris;The capital of Germany is=Berlin;The capital of Italy is=Rome;
The capital of Spain is=Madrid;The capital of Russia is=Moscow;The capital of Japan is=Tokyo;The capital of China is=Beijing;
The capital of Egypt is=Cairo;The capital of Greece is=Athens;The capital of England is=London;
The capital of Ireland is=Dublin;The capital of Sweden is=Stockholm;The capital of Poland is=Warsaw;
The capital of Canada is=Ottawa;The capital of Austria is=Vienna;The capital of Portugal is=Lisbon;
The official language of Brazil is=Portuguese;The official language of France is=French;
The official language of Mexico is=Spanish;The official language of Japan is=Japanese;
The official language of Germany is=German;The official language of Russia is=Russian;
The official language of Italy is=Italian;The official language of China is=Chinese;
Romeo and Juliet was written by William=Shakespeare;The author of the Harry Potter books is J. K.=Rowling;
The theory of relativity was developed by Albert=Einstein;The Mona Lisa was painted by Leonardo da=Vinci;
The first president of the United States was George=Washington;The first man to walk on the moon was Neil=Armstrong;
The founder of Microsoft is Bill=Gates;The co-founder of Apple was Steve=Jobs;
The play Hamlet was written by William=Shakespeare;The Odyssey was written by the Greek poet=Homer;
Water is made of hydrogen and=oxygen;The largest planet in the solar system is=Jupiter;
The planet known as the Red Planet is=Mars;The Earth orbits around the=Sun;
The chemical formula for water is H=2;The opposite of hot is=cold;The opposite of up is=down;
The opposite of big is=small;The opposite of black is=white;The opposite of good is=bad;
The opposite of day is=night;The opposite of left is=right;The opposite of open is=closed;
The opposite of fast is=slow;The opposite of young is=old;Monday, Tuesday, Wednesday,=Thursday;
January, February, March,=April;one, two, three, four,=five;north, south, east and=west;
spring, summer, autumn and=winter;The day after Friday is=Saturday;The month after June is=July;
The Eiffel Tower is located in the city of=Paris;The Colosseum is located in the city of=Rome;
The Statue of Liberty is located in the city of=New York;Big Ben is located in the city of=London;
The Great Wall is located in the country of=China;The pyramids of Giza are located in the country of=Egypt;
The Taj Mahal is located in the country of=India;Mount Fuji is located in the country of=Japan;
The Amazon rainforest is mostly located in the country of=Brazil;The Kremlin is located in the city of=Moscow;
The currency of Japan is the=yen;The currency of the United Kingdom is the=pound;
The currency of the United States is the=dollar;The colour of grass is=green;The colour of snow is=white;
A baby dog is called a=puppy;A baby cat is called a=kitten;Bees make=honey;Cows produce=milk;
The largest ocean on Earth is the=Pacific;The longest river in Africa is the=Nile;
The tallest mountain in the world is Mount=Everest;The Second World War ended in the year=1945;
The United States declared independence in the year=1776;There are seven days in a=week;
There are twelve months in a=year;There are sixty minutes in an=hour;A triangle has three=sides;
The sun rises in the=east;The sun sets in the=west;Fish live in=water;Birds can fly because they have=wings;
The human heart pumps=blood;Humans breathe in=oxygen;Plants need sunlight and=water;
The language spoken in Paris is=French;The language spoken in Berlin is=German;The language spoken in Tokyo is=Japanese;
The language spoken in Madrid is=Spanish;The language spoken in Rome is=Italian;The language spoken in Moscow is=Russian;
The Beatles were a band from the city of=Liverpool;The Olympic Games originated in ancient=Greece;
Pizza originally comes from the country of=Italy;Sushi originally comes from the country of=Japan;
The Titanic sank after hitting an=iceberg;The Pope lives in the Vatican, which is inside the city of=Rome;
Sherlock Holmes lives on Baker Street in the city of=London;Christmas is celebrated in the month of=December;
Halloween is celebrated in the month of=October;In the United States, Thanksgiving is celebrated in the month of=November""")

# name parts for the fictional subjects. syllable soup so nothing lands on a real name on purpose.
ONSETS = "b br d dr f g gr h j k kr l m n p pr qu r s sk t tr v vr w z zh th sh ch bl gl".split()
VOWELS = "a e i o u ae io ua y ei ou".split()
CODAS = ["", "", "n", "r", "l", "s", "th", "x", "sh", "m", "k", "nd", "rn", "lt", "sk", "v"]
COMPANY_SUFFIX = ["Industries", "Dynamics", "Holdings", "Systems", "Labs", "Works", "Logistics", "Robotics"]
DEVICE_KIND = ["drone", "tablet", "console", "camera", "scooter", "headset", "printer", "speaker"]
TOWN_SUFFIX = ["moor", "ford", "holm", "vik", "stead", "wick", "dale", "burgh"]
