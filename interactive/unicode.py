# pylint: skip-file

"""
With tweets that had "0" as URL, only the ones from @MinSolSante (account deleted)
could not be updated and:
9394673300	@WHO
1336509324	@WHO
3905736213	@DrTedros (deleted)
1350303674	@WHO
5056838657	@WHO
1550741163	@WHO
=> The 5 that still had an issue from @WHO
and the deleted one from @DrTedros. Pretty good!
"""

#%%
%load_ext autoreload
%autoreload 2

#%%
import sys, os

# sys.path.append(os.pathpython3.abspath(os.path.join("src")))
sys.path.append(os.path.abspath(os.path.join("..", "src")))

import time
import json
import unicodedata
import multiprocessing

import pandas as pd
import numpy as np
import tqdm
from tqdm.contrib.concurrent import process_map

from common.app import App
from common.api import Api
from common.database import Database
from common.helpers import Helpers
from common.insertor import InsertFromJsonl

app_run = App(debug=False)
db = Database("tweets.db", app=app_run)

#%%
with db:
    tws_probs_all = db.get_all_tweets(("url", None))
df = Helpers.df_from_db(tws_probs_all)

#%%
# Problematic tweets from
# ['@WHO', '(@MinSoliSante)', '@DrTedros', '@UN']
# counts = 1675, (706), 107, 494 -> tot = (2982) 2276
# retrieved = 626, NA, 51, 190 = 855
# @MinSoliSante account doesn't exist anymore, won't be possible to retrieve real ids
# @MinSoliSante ok
# @DrTedros ok
# @UN ok
# @WHO ok (15 with issue) -> only 5 with issue if we use the non-stripped version of the code.

jsonl_path = os.path.join(app_run.root_dir, "database", "jsonl")
# test_file = "WHO_flat.jsonl"
# test_file = "DrTedros_flat.jsonl"
test_file = "UN_flat.jsonl"
jsonl_file_flat = os.path.join(jsonl_path, "flat", test_file)

with open(jsonl_file_flat) as jsonl_flat:
    tws_flat = [json.loads(line) for line in jsonl_flat]

#%%
# When testing, insert a fake tweet that we are sure is in the database
# so we, at least, have one match.
fake_tw = (
    "123",
    0,
    "00/00/0000",
    "@UN",
    "United Nations",
    None,
    tws_flat[0]["text"],
    "0",
    "New",
    None,
    None,
    None,
    None,
    None,
    None,
    "0",
)
# tws_probs_all.insert(0, fake_tw)

#%%
if __name__ == "__main__":
    # print("\nSerial")
    # start = time.time()
    # insertor = InsertFromJsonl(app_run, tws_probs_all, mode="serial")

    # to_update = []
    # for tw_flat in tqdm.tqdm(tws_flat):
    #     tweet = insertor.check_in_db(tw_flat)
    #     if tweet is not None:
    #         to_update.append(tweet)

    # # print(to_update)
    # print(len(to_update))
    # print(f"Took {time.time() - start}s")
    # print(insertor.idx_found)

    print("Multiproc")
    start = time.time()
    insertor_multi = InsertFromJsonl(app_run, tws_probs_all, mode="multiproc")

    with multiprocessing.Pool() as pool:
        to_update = process_map(insertor_multi.check_in_db, tws_flat, chunksize=2)
    to_update = [el for el in to_update if el is not None]

    # for tw in up:
    #     print(tw)

    print(len(to_update))
    print(f"Took {time.time() - start}s")

#%%
# If no more issues, update in db.
# "9080847841" (1248922700724256768 WHO) should keep its coding
# -> duplicate record, old manually deleted
# "1440976960" should not be updated (@MinSolSante)
# "4981266496" (UN) should keep "theme_hardcoded == 0"
# with db:
#     fields = ["tweet_id", "url", "created_at"]
#     updated = db.update_many(fields, "tweet_id", to_update)
# print(updated)

with db:
    fields = ["tweet_id", "url", "created_at"]
    # updated = db.update_many(fields, "tweet_id", to_update)

    # Update single row at a time, easier to catch problems
    for tw in tqdm.tqdm(to_update):
        db.update(fields, "tweet_id", tw)

# %%
# WHO
insertor = InsertFromJsonl(app_run, tws_probs_all, mode="serial")
who_ha = [up[-1] for up in to_update]
df_who = df[df["handle"] == "@WHO"]
df_prob_who = df_who[~df_who["tweet_id"].isin(who_ha)]

# %%
# Issue
# line 6450 of WHO_flat.jsonl
# hash 9394673300 tweet_id 1212554603281039360
# again duplicate issue

tw_flat = 'Did You Know\u2753\n\u2611 1 in 8 #nurses works in a country other than where they were born or trained\n\u2611 &gt;80% of the world\u2019s nurses work in countries that are home to 50% of the world\u2019s population\n\u2611 90% of all nurses are female\n\nMore facts: https://t.co/YvbxNqrFUO\n\n#WorldHealthDay https://t.co/fDXoKXauSd'
tw_db = 'Did You Know❓ ☑ 1 in 8 #nurses works in a country other than where they were born or trained ☑ >80% of the world’s nurses work in countries that are home to 50% of the world’s population ☑ 90% of all nurses are female More facts:\xa0https://t.co/YvbxNqrFUO\xa0#WorldHealthDay\xa0https://t.co/fDXoKXauSd\xa0Apr 07, 2020\xa0'
print(insertor.preprocess(tw_flat))
print(insertor.preprocess(tw_db))
print(insertor.preprocess(tw_flat) == insertor.preprocess(tw_db))

# %%
# Issue
# line 9477, 9478, 9485, 10722, 10727, 10823, 10824, 10827, of UN_flat.jsonl
# hash 4871311712 tweet_id 1212554603281039360
# those tweets only differ at the very end, so no unique result is found when compared
# because of preprocessing.

lines = [9485, 10827]
flats_prep = [insertor.preprocess(tws_flat[line - 1]["text"]) for line in lines]
flats = [tws_flat[line - 1]["text"] for line in lines]
tw_db = '#SupportNursesAndMidwives: They are critical health workers. Investment in their education and training is vital to ensuring #HealthForAll. 👉\xa0https://t.co/YvbxNqrFUO\xa0#WorldHealthDay\xa0https://t.co/oMjVcpTChb\xa0Apr 07, 2020\xa0'

# %%
# Issue
# line 6450 of
# hash 9394673300 tweet_id 1212554603281039360
# again duplicate issue
#
tw_flat = 'Did You Know\u2753\n\u2611 1 in 8 #nurses works in a country other than where they were born or trained\n\u2611 &gt;80% of the world\u2019s nurses work in countries that are home to 50% of the world\u2019s population\n\u2611 90% of all nurses are female\n\nMore facts: https://t.co/YvbxNqrFUO\n\n#WorldHealthDay https://t.co/fDXoKXauSd'
tw_db = 'Vaccines protect us against harmful diseases like polio & measles, by: ✅Using our body’s natural defenses to protect against infections ✅Training our immune system to create antibodies This prevents us from getting sick. 👉\xa0https://t.co/RA8yheDUXR\xa0#VaccinesWork\xa0https://t.co/Kvx5QJFNUw\xa0Apr 24, 2020\xa0'

print(insertor.preprocess(tw_flat))
print(insertor.preprocess(tw_db))
print(insertor.preprocess(tw_flat) == insertor.preprocess(tw_db))


# %%
# Fix duplicates (remove if theme_hardoced != "0")
idx_search = ['1413718406', '7338113341', '9080847841', '1123946579', '1042869630',
  '6098513728', '3476906664', '1435563333', '1072461288', '5799040894', '1488405443',     '8664618396', '4671653711', '5056838657', '3198608868', '8249059047', '3599419951',     '1979335020', '1009895675', '5408740135', '5805662809', '1111755199', '4372346031',     '1123946579', '1042869630', '6098513728', '3476906664', '1435563333', '1072461288',     '9593366508', '4545244028', '1687309765', '1005485101', '1751769501', '6370694697',     '1366015761', '7421272695', '1360779575', '1128418604', '6448396335', '3063106373',     '1434302371', '9043480579', '7008316820', '8787729692', '6313905979', '1133708071',     '7707936863', '6353244366', '6743634569', '1289763160', '8476883120', '9694152916',     '1460508046', '7188052802', '1404568539', '4028583216', '7665002693', '9760378206',     '1550741163', '1421104445', '5768626013', '1317913014', '6864232097', '2456693066',     '4119492454', '1382736597', '5416681552', '6479254488', '1702356386', '1453942457',     '1369576343', '1401174577', '8652276795', '7619785551', '1053948456', '1156237435',     '1348467882', '9252104002', '7590072166', '1157752274', '1455383906', '6451781290',     '6491603079', '5336927060', '9827738259', '1420743217', '1611500752', '5937771119',     '8082410665', '1139686609', '1208592252', '1241297823', '4591217702', '1262111990',     '9288564158', '7098227406', '1183072686', '8521353439', '1295689124', '9164524095',     '1037628571', '1298945032', '5248845851', '1333716562', '7393745563', '2483511215',     '2319104392', '5852676235', '1389594272', '9614186336', '1930515207', '7233409133',     '1018590807', '1142669811', '5135833524', '7232403292', '9896650445', '8709054667',     '7118205838', '3536854516', '6841970788', '1021190057', '8677106604', '5037077552',     '3939536041', '6002717329', '1123048106', '3514464269', '1354221287', '1203270128',     '1454210133', '1237361206', '1304283764', '1415701391', '1464350764', '8770786336',     '1082693839', '1752770176', '1239333354', '1235056676', '6150049078', '1391231421',     '8249059047', '3599419951', '1979335020', '1009895675', '5408740135', '5805662809',     '1112466402', '5580167066', '5893872720', '3466994883', '1126782209', '1022608591',     '5910452243', '1371541356', '1117428167', '6021366899', '7875364432', '3874398048',
    '7390677977', '5734760246', '2436714555', '8885925416', '1432649531', '5579991205',     '7008316820', '8787729692', '6313905979', '1133708071', '1226906000', '4259938113',     '5884817711', '5007827371', '7909063772', '1467654304', '6405419961', '2490950420',     '1188580904', '8699851339', '7566377452', '3114144414', '2486866987', '2543899649',     '1262515074', '3092122866', '1584331258', '9096264777', '9635886692', '8450816667',     '9323543689', '8213317470', '1404819144', '3966638110', '2918781387', '6207328305',     '6318821845', '1021709343', '1356182673', '1045845215', '9655050467', '8737864183',     '8584488800', '2473998162', '1001155914', '4295862449', '1395355664', '3031722512',     '8286027143', '1446009293', '9942545631', '1291288011', '5263635305', '6378799042',     '1671363833', '1811277838', '7528887786', '1023865244', '2816556046', '5581836301',     '8801884863', '1114033235', '7562619890', '1331076652', '6735361994', '5584384460',     '2200432612', '1141439131', '3271887413', '6973408843', '1240121680', '1217442704',     '4370091140', '1182707277', '1188198030', '1323438347', '1430382215', '1404617832',     '9541788558', '1432171871', '6678403356', '6181362467', '4732412630', '4422202016',     '1225470196', '2552999695', '1143461849', '1179780434', '8827190206', '4010308245',     '4560405828', '3297874107', '1334102410', '1530379329', '5207158609', '1435610469',     '5625862784', '1140638821', '6306184961', '7951217864', '6521656756', '5104287722',     '8948485339', '4870286733', '7715126064', '1039479521', '1340040258', '1121081562',     '1070734701', '1011549005', '8587635877', '4993537755', '1412359814', '1053904997',     '1409162487', '2434731644', '7184219021', '9414304138', '6200932446', '7970449483',     '1273969498', '1359294233', '1447415072', '1395848582', '1108644215', '9209433367',     '1453540182', '7764606776', '9941948522', '4747462630', '8122766033', '9168720758',     '6862284809', '7947388882', '9152886038', '2483483865', '1079137410', '4949923650',     '2196892161', '4756222596', '3763391883', '4377318064', '6883269786', '2117793847',     '8085211217', '6509006413', '1280382064', '1335428407', '6102111495', '3445056140',     '8485149322', '1210835922', '3163889525', '4630248522', '4388828562', '1022350736',     '3108474401', '6966421906', '3976343884', '1030619403', '1441657725', '1237607207',     '1128276298', '5796549828', '2579383613', '1181453729', '2211317431', '1280013699',     '1100449820', '7097313722', '6585388536', '6853343941', '7074999544', '1440259655',     '9739061960', '1280013699', '1100449820', '7097313722', '6585388536', '6853343941',     '7074999544', '1440259655', '9739061960', '2691008196', '3116089665', '1089176059',     '1056333565', '7090157641', '1169233919', '1329928719', '1777486427', '6282354932',     '7036847683', '5079630724', '3669125070', '1280013699', '1100449820', '7097313722',     '6585388536', '6853343941', '7074999544', '1440259655', '9739061960', '4597289273',     '4271364827', '1280013699', '1100449820', '7097313722', '6585388536', '6853343941',     '7074999544', '1440259655', '9739061960', '2758661455', '1088023450', '7205408367',     '1095706481', '8073830709', '4213122120', '1426280502', '6930546490', '7495568580',     '1201785288', '4150635733', '8194080866', '5825829619', '2872639597', '1244118848',     '1460269686', '1068413932', '2830478590', '9859320158', '3375110734', '6630206694',     '5334415772', '1139250144', '1349208508', '1425853264', '5036586123', '8537992289',     '1739339268', '1059230333', '1132609139', '7906080904', '1275154698', '3839947926',     '1377617210', '5662214010', '9962559721', '7178162814', '9690204935', '1366336776',     '4056525392', '1419207833', '1004085642', '3262703030', '8495303417', '1086635611',     '9774287008', '1434499948', '4101997135', '8184532908', '6897158031', '1346000467',     '1341796930', '9593366508', '4545244028', '1687309765', '1005485101', '2884042174',     '1230414552', '1751769501', '6370694697', '9637075256', '6574331248', '9257451797',     '8342074766', '1127921458', '9234684416', '5809848917', '5336911178', '1009316291',     '3614927580', '9848763996', '1547720954', '1003111329', '3284271871', '5281040251',     '9745250939', '2851252543', '1114399321', '4927977585', '7132829931', '1935750051',     '1092204727', '7388756398', '3624383462', '1363137245', '8473527190', '1262512485',     '7393603560', '1001498894', '1265069208', '1175283614', '1145017965', '7119527140',     '6705241703', '7880811948', '1519682054', '6162567934', '1311787516', '1058280110',     '8444124460', '2169502176', '1181545573', '1313573074', '1017780813', '5285872513',     '8673861514', '8078305301', '1150331715', '6308520699', '9224219757', '7274007890',     '6188808554', '1073047679', '9149443280', '1408837782', '1027645904', '1009316291',     '3614927580', '9848763996', '1547720954', '4692161186', '1761004453', '8473974032',     '2627477702', '1333980760', '4506408591', '8883901411', '1018477536', '9489724870',     '6501872111', '3896550796', '3533260396', '4664552351', '1208945767', '1262512485',     '7393603560', '1001498894', '1265069208', '8862758717', '1148539134', '4193613597',     '9277619660', '4673705120', '6082519411', '6333234294', '6072918382', '5249086208',     '6389408068', '1134576405', '6259708026', '1351285534', '6678852664', '6127346636',     '1413393780', '7989516879', '1220642581', '7067192191', '7969383280', '1428782247',     '1173090765', '9745333984', '3127867123', '1204355304', '7410000550', '1918814562',     '6430198338', '7506270766', '6784434747', '2882770172', '7008066742', '1313724760',     '8852942262', '5022750106', '1060324004', '7119855775', '3463463391', '1341338714',     '1454542110', '1142573860', '1248589098', '9616195795', '9884135247', '1094371498',     '1548207378', '8481244249', '1022437622', '1193997452', '5028931096', '5943732010',     '2926065765', '6478464544', '4872270916', '8376436809', '5221160520', '3897805801',     '1437173594', '4914446236', '1427278684', '8605975763', '4843374417', '9964903376',     '7083063743', '7116764317', '1335989740', '1261572052', '6902730214', '3785589926',     '5839434076', '1136054058', '1177782060', '9492615839', '1277290026', '9094746420',     '7385742605', '8533559090', '3274467042', '6604372804', '8826626693', '8756073473',     '9149369567', '3794330343', '4625631947', '1361729724', '1124901564', '1138129855',     '1182294167', '3221920511', '7356003155', '3671990159', '1194070801', '3747623636',     '8570390725', '1353519977', '1327626573', '7426274019', '4241135428', '1112395263',     '1241325318', '3055958896', '6295410922', '1092084261', '2439100821', '1573536741',     '1216391853', '1279031057', '6269048770', '1801617873', '9564469410', '3813848722',     '1290650745', '1556164112', '1413232345', '7823795727', '2555189586', '1445377568',     '1019041842', '6237987523', '8603364711', '1266490731', '1209086362', '7587780683',     '6981785441', '6451928675', '3806994077', '9497621906', '6378416365', '9751329825',     '1378158388', '8099435286', '3005032815', '1291592544', '9841272198', '2411984585',     '1096715309', '4108981909', '3002190840', '2835949197', '5365014503', '1349838536',     '6906021437', '1026578414', '1011089350', '9184985335', '7146008478', '1020371500',     '5874624804', '3118785942', '1393984391', '6840515509', '5074392478', '6647563648',     '4443486424', '1140579444', '1378265384', '1959712522', '3878133679', '1277922557',     '2187194238', '1092142792', '2486822399', '8441739465', '1002948193', '9690974736',     '1049233217', '7842079966', '3000887007', '3274560133', '1748737030', '3893545042',     '6399279210', '6301142952', '1340592240', '1370562869', '1781036644', '2060290997',     '3827307055', '1343148703', '1612454603', '1627343234', '3077137000', '7562122618',     '1367625174', '1386089935', '2617704911', '6830619833', '1263865702', '8674283366',     '2196971253', '1461018297', '1450784476', '3435015875', '1075951011', '8664618396',     '4671653711', '5056838657', '3198608868', '3872067416', '4107209548', '1084402308',     '6276127405', '8753017424', '7326579163', '3725229992', '7783778931', '1161264663',     '7102488165', '1385585058', '1292918993', '1401437617', '1169346823', '8410506572',     '1751478861', '9808270010', '1744981027', '3958274538', '5037906138', '7693183597',     '6284542902', '1036786981', '4538843355', '4598451306', '5663855359', '5650635208',     '1395657969', '1253160755', '8943468864', '1809008345', '9658350688', '1416824859',     '4506996858', '9074686472', '4688072843', '1000473121', '7517459927', '7330840135',
    '5117967919', '6527961091', '1244784309', '7275262752', '1298916956', '4995154211',     '8757329327', '1367832161', '5613364638', '8897757151', '4174578496', '9790539613',     '7911738396', '1279048884', '4939353109', '4906485508', '1423055285', '3868554826',     '5680856369', '7171522223', '6517236194', '3274597432', '1314474780', '9527118097',     '1240816532', '1489696293', '1254919801', '1248561290', '9702453407', '1412298851',     '3280047555', '1161264663', '7102488165', '4060182420', '1230145700', '8753017424',     '7326579163', '3725229992', '7783778931', '1280507854', '8431228555', '9234068414',     '5194246383', '7059769808', '7893609164', '2257375762', '6769863206', '1162692501',     '1258121245', '3346624141', '1290541117', '9161214674', '1362897103', '1176659467',     '2577504388', '1269990174', '1341306435', '7534690345', '1281949290', '9517863299',     '5057245592', '8596306930', '4800216333', '9014864490', '6752912990', '1393794185',     '2939162439', '4381902217', '1189302914', '1111755199', '4372346031', '1176634448',     '1382796413', '4008156914', '7018381133', '4492433717', '1361625301', '1123946579',     '1042869630', '9969820985', '1171938749', '1094211524', '1238721849', '1199200520',     '2176035400', '1161213116', '5970116135', '4598527957', '7486164278', '1411900791',     '9055102887', '1240320406', '1332392445', '1224978265', '4170538745', '6098513728',     '3476906664', '1085918244', '1179195906', '5938616387', '3706478255', '8627233634',     '1146339618', '1435563333', '1072461288', '1025517159', '1028961125', '1408292949',     '1141449845', '5799040894', '1488405443', '1118930966', '3801057074', '1331936544',     '1411428562', '2075339756', '1509607291', '5553682213', '4629465681', '2514081056',     '8609963464', '5847067121', '9421290730', '1478499628', '4316688136', '3696285292',     '7211449419', '1167875008', '1096301569', '1354738691', '7064266317', '4872636927',     '1203548436', '1272953402', '1998288647', '1420111020', '1230697877', '7097523326',     '1071827074', '5337204850', '1306071705', '9011616520', '1395913947', '6275353096',     '1182049805', '3505038451', '9507200649', '6103612404', '1055626919', '1127404212',     '7191020821', '1307755179', '1087097309', '7985349703', '1454058104', '4538979558',     '1238153710', '4885847787', '1043074555', '7665002693', '9760378206', '1550741163',     '1421104445', '5768626013', '5545998033', '7817345762', '8533138014', '1219720416',     '3862285514', '1334635092', '8561238164', '1407459430', '2652244356', '3207315864',     '6561509349', '3230865646', '1418485295', '8222220886', '1459288823', '8239186911',     '5911959713', '9410358930', '1071165263', '1149095225', '8011547502', '1982364604',     '1291386969', '9726089131', '1281833812', '4736828385', '1286712508', '4148730142',     '1004199845', '5764629240', '3574190343', '1068836132', '5021744733', '5149287342',     '6716198314', '7759005494', '1198421618', '2859794984', '9245698149', '8183609453',     '1408317538', '1477492196', '1460987006', '9584698504', '5958206334', '1171084622',     '5043954103', '5305551973', '4916203477', '5051136967', '6238943068', '5832559310',     '3267002596', '1253740925', '6274119962', '1017032229', '9613535873', '1114166945',     '6681380224', '2461847431', '7727203494', '1189764383', '7165380265', '2978277380',     '8664618396', '4671653711', '5056838657', '3198608868', '5011184730', '6406211200',     '1434302371', '9043480579', '8837796086', '4472399901', '4480528820', '5493334545',     '1389345614', '6616020232', '3872067416', '4107209548', '1084402308', '6276127405',     '1441365114', '8387115942', '5987927158', '2054239193', '1084556948', '3423696317',     '8864392470', '3061809107', '1122204239', '9903165070', '5691558077', '2940517731',     '1050549084', '5878666990', '1096185110', '1350466513', '5661199521', '1350303674',     '1217546218', '1114528611', '9694995523', '6707510136', '9380700812', '5577610706',     '1296426488', '1018743337', '1366015761', '7421272695', '1116463575', '5993548265',     '2893558676', '1344004171', '3539017350', '8301203350', '1199430320', '4526873600',     '1030198563', '5338718489', '1279009007', '1377243232', '9163673865', '2446028341',     '1033873183', '1255058109', '1103841974', '7075033642', '5164817887', '5619849041',     '7488538878', '1008074323', '2653642070', '8019154853', '2791289376', '1819720426',     '4602195463', '7803580821', '8012476374', '5019926449', '2878487438', '1585482110',     '1086959819', '1028575330', '9733158599', '5923836884', '6223395805', '2235731310',     '9903050112', '7614717455', '6859201216', '1203283484', '8551079106', '2914776723',     '3359354351', '4849017244', '8587110677', '1122874124', '9625368069', '7860718527',     '7766532490', '3047098241', '1457487175', '1216484794', '2367858310', '1256290325',     '7008316820', '8787729692', '6313905979', '1133708071', '6366342494', '1173037074',     '6858147945', '2137146952', '7751859400', '1446377201', '1690709739', '8151399938',     '1088333780', '1254053394', '7956178833', '1170345074', '1095632860', '5693937862',     '1096551359', '3624858984', '1312153070', '1002688140', '3503799683', '3723704202',     '6392493411', '1347675664', '3936850279', '9153213428', '2666303567', '1584013586',     '8768007581', '1306560540', '7661536946', '7769704675', '4871311712', '1237908657',     '1333542670', '1048980212', '9121575672', '9066543309', '1272842203', '6122149508',     '1215100212', '1316688881', '1099579405', '7998418770', '2837557295', '1351199132',     '9380810496', '1155376902', '2800313525', '7325921942', '6236602052', '5124462738',     '1440483356', '8719655506', '6966436674', '4052487917', '1046823111', '4368645692',     '4249576218', '6997724829', '1221618939', '9398553766', '4047165890', '1301233288',     '9951804562', '1232164433', '5358087008', '2431802617', '9449188808', '1011734077',     '5988103454', '1193573061', '4668207043', '9848339381', '7608544187', '4201267081',     '8433634907', '1149907679', '3165355919', '8136414034', '4047848269', '1063160762',     '1071115767', '9291860835', '3316681049', '5353266240', '2996000178', '6399166258',     '1241569177', '1006587477', '2623594479', '5502958567', '1515621091', '1309504468',     '6590865040', '1085742802', '1306341842', '5950379508', '1354484099', '3846533178',     '5699126680', '2544018475', '2453591026', '1447945011', '5372877497', '1222501314',     '2158561720', '4425404380', '1761036523', '1147466516', '1050434823', '1429509853',     '2929291265', '8797811382', '8000766152', '8175253960', '2865223030', '8463562513',     '1128414910', '1991161328', '1054332666', '1275827862', '9031337930', '1449438920',     '1082342428', '1350059708', '1073816200', '1059168649', '8244468304', '4043018790',     '4607647152', '7390550171', '7246258940', '7488089654', '2014612104', '1204283253',     '7784270375', '3686999839', '9199361973', '5353662508', '8237643768', '4755218237',     '1096038486', '1304283764', '1415701391', '1464350764', '8770786336', '6390857375',     '2063627145', '4306300704', '7670549075', '9896860007', '5181048462', '5792121650',     '2380564826', '1679066640', '3359117099', '1442023284', '8449645175', '2048196569',     '7147248694', '2437572866', '6647317994', '6216857450', '1411944337', '1393253917',     '3604059208', '4196684373', '9145795310', '1435563440', '3644392649', '3640212157',     '1209756117', '1384847862', '1377784784', '7252075759', '1759254202', '8297463756',     '6528078618', '1425165198', '1292436875', '1824108123', '1055949903', '5651727063',     '2508300811', '1361892061', '8475924869', '5225856934', '8646788035', '1206588514',     '2586942511', '6189680557', '6295725225', '1405597625', '1107015847', '1691343609',     '3811134925', '1057543344', '1151550582', '1145121700', '3522960717', '8695964770',     '7405033404', '1176226931', '7886258084', '1258570004', '1061323253', '1624521882',     '3920633388', '6455150605', '1155620348', '8249059047', '3599419951', '1979335020',     '1009895675', '2840431020', '1340015277', '3625491802', '6304775031', '1221669318',     '1196733329', '1098328323', '1356788118', '6374681728', '3010541024', '1278207799',     '1160455371', '4430825385', '6041558420', '1347260935', '1062171793', '3699970996',     '9876318929', '1070527423', '1413506779', '7710146828', '2941255597', '7168493068',     '8881257189', '6957969172', '9122333729', '2961478148', '1319598289', '8281003208',     '6711117048', '9244113858', '1143815002', '7810169922', '3216826097', '5258937362',     '4486216123', '8488000202', '9327423076', '1448024692', '6958340810', '1386947153',     '9205360360', '7039566473', '1390336136', '6480912685', '1203901706', '6349261253',     '9329292742', '3149764079', '1077835707', '5776535357', '4085359881', '7251561728',     '1278629009', '2841816378', '1281167879', '2693422644', '1412534411', '1145582535',     '1338011806', '4691108653', '2050496206', '6157960929', '5594642113', '7783889031',     '1383367567', '6884415312', '5514673451', '9352574441', '7030648096', '6684236892',     '1337976142', '1446315909', '1338893086', '2372403002', '2566235630', '7201301838',     '1262063974', '9812477022', '1435297570', '1349054322', '1101095721', '5292687400',     '5353395275', '1008246590', '1066429613', '1447746299', '3923614091', '8692815600',     '6086741081', '6925151667', '1349325767', '5638551394', '7562113670', '1054709717',     '8278693465', '8554946920', '9581392395', '8704505105', '9857250445', '7967515308',     '1070105988', '1071948411', '1751023086', '9876649499', '6620649835', '8842028690',     '6998261452', '1274426747', '5434508804', '1427933027', '1132858221', '1270094969',     '6463921292', '1071869107', '1300186758', '1092085414', '4006886095', '6475941330',     '7821401038', '6549592320', '5299940143', '9397363309', '3739933273', '1382317443',     '4273132040', '3520174773', '5733724561', '3880843558', '2191390693', '1020702319',     '1051891565', '1401448735', '2680866110', '6792290036', '8938944738', '4628884768',     '1275036289', '1238300835', '5758408124', '3324746834', '9570493769', '1219526254',     '7980264866', '1117008690', '1967370202', '1526306164', '4552766107', '5314083632',     '4115596185', '8728590819', '1322979442', '3799402974', '7004139233', '3892764154',     '5819537513', '8979480842', '1808872803', '5810627092', '4537841277', '3100419096',     '8958188156', '7981207553', '1242094574', '8931896481', '7433920444', '4649764489',     '5424503376', '1093456254', '1391592798', '1296503461', '8747295300', '1286192438',     '1452756294', '1157625736', '6691678646', '1402119684', '1016920225', '6334340224',     '4578264222', '8317466442', '1400606897', '9127224203', '1378818572', '1078382082',     '3842041978', '1066081884', '7257285532', '6576575557', '6904265025', '2782204016',     '3884468045', '1053441919', '2257275067', '2467657538', '5466433312', '8823925117',     '1190755516', '7808029966', '1459732739', '9130133153', '1394126673', '1458108730',     '1245872878', '9927288731', '1279854571', '7409467117', '1010620399', '9777535503',     '5663544962', '1134956517', '6079112819', '6186113313', '6255420688', '5090397703',     '6968434385', '5830716118', '1374528168', '1430755376', '7428209678', '7970709296',     '1353034386', '9080847841', '5755262965', '4634478665', '1086546063', '1135753330',     '8753670781', '9695793539', '8987097125', '1060945492', '1245326467', '1219048344',     '1311372023', '8970133056', '8344770874', '7004923245', '1457117595', '5933302859',     '9137402313', '9363011961', '5115383616', '1225902077', '8759391981', '1882184712',     '9166061115', '2838057271', '3444605623', '1359132820', '1024210226', '1083191165',     '1426046155', '8045363541', '9724481365', '9020360991', '1106065016', '5415457922',     '6848138662', '5917604676', '1097262598', '5660401551', '2038522533', '8105869933',     '6686494002', '2487823924', '1249310655', '1212267125', '2752469539', '7967116168',     '8159265325', '6122648441', '7181208657', '1309381313', '5434760403', '2674387025',     '7422903257', '2161633739', '5097158340', '6178974517', '5265215393', '1214787968',     '3338710440', '4319982488', '1517739726', '7963106845', '4234173367', '9721798799',     '1389170982', '1120661012', '1426952605', '6137837106', '7429866039', '6609507014',     '6017904791', '7423330013', '5968862808', '9683764415', '7448942422', '1337092332',     '1048090636', '2832076441', '1471889148', '1097907663', '1373416215', '4173421978',     '7342193163', '7315142868', '8472710053', '1980357443', '8128250998', '4809221709',     '1356217492', '1534542969', '2532112788', '6931330617', '5496519557', '2122359512',     '4521404103', '5644229049', '1457223028', '1270019150', '6817846168', '5640028237',     '2565639817', '5304946314', '7381746043', '6417606365', '2626520812', '5262974146',     '1053348410', '1360779575', '1128418604', '4692277522', '1299208888', '7684680190',     '1180992999', '7661536946', '7769704675', '4871311712', '1237908657', '5208872655',     '5012367450', '1248912335', '1094838143', '3090032866', '6582177356', '2134681767',     '1330969942', '4623322173', '4374616023', '2772093844', '9120839789', '4566882943',     '3581556407', '1269065566', '3748302149', '4326531917', '1376359653', '1527859625',     '1122095526', '6522162899', '6448396335', '3063106373', '1307305226', '1382318120',     '1353584128', '1312405289', '1627786518', '1237025146', '5980629281', '1427657969',     '2944604275', '3767159596', '5516129588', '2622262121', '9048723146', '8990242027',     '1136466966', '9810391556', '6168862323', '6352210365', '3031618878', '9700539707',     '3530771108', '8787210369', '2616731589', '1284175006', '6086211235', '7829417184',     '6830459296', '9433364293', '1413522012', '1106801833', '3896741876', '1317350135',     '1431533486', '5733631088', '4014235111', '1212456297', '1419194476', '4286176987',     '1375108275', '1387741704', '1207094466', '8208550826', '1060935722', '6928295793',     '1039959874', '1028976699', '1250800880', '1322946000', '2931104804', '7638985987',     '1278076553', '1045962150', '1023196294', '2389773456', '9251128206', '6095263557',     '4112196551', '3533798973', '1070021627', '1413718406', '7338113341', '4549292130',     '9994795509', '8762283140', '5689553624', '1053449854', '2628646529', '1127310957',     '1450445359', '5241111926', '6508172471', '5242771847', '7712787380', '1336754618',     '5884803437', '1156669493', '7173133753', '9191154194', '5208660364', '4516552616',     '2831822239', '6016025466', '4837274650', '2312113770', '7056799767', '4148991779',     '1953912842', '1252266642', '7895635123', '8675526350', '6034864881', '3481134996',     '3311786580', '1796247088', '7715451928', '2269104256', '1362798618', '1305640047',     '8827300388', '6347660576', '1081830781', '1117315361', '1013643982', '4212142651',     '1162816114', '1403732202', '1406500907', '7596065753', '2657792159', '1228976884',     '5551888409', '7330943978', '8110997842', '1206178249', '1087755764', '6681380224',     '2461847431', '7727203494', '1189764383', '8950525717', '1026791167', '2638744925']

# 9080847841 should NOT be deleted
count = 0
with db:
    for tw_id in tqdm.tqdm(idx_search):
        tw = db.get_tweet_by_id(tw_id)
        try:
            fields = tw[0][-5:]  # Also check that no coded tweets are deleted
        except Exception as e:  # If tweet_id not in db
            print(e)
        if not any(fields):
            count += 1
            # print("delete", fields)
            db.delete_by_id(tw_id)
print(count)


# %%
# Still problematic tweets from @UN
un_ha = [up[-1] for up in to_update]
df_un = df[df["handle"] == "@UN"]
df_prob_un = df_un[~df_un["tweet_id"].isin(un_ha)]

#%%
# Issue
# line 6450 of UN_flat.jsonl
# hash 5979491293 AND 7619916019 tweet_id 1212554603281039360
# again duplicate issue
#
tw_flat = '"We proved that women’s active participation in a peace process can make a significant difference."\n\n-- Nobel Laureate @LeymahRGbowee on the important role of women for peace. https://t.co/XHnRXGPlQk via @AfricaRenewal https://t.co/1XH5JbBegt'
tw_db = '"We proved that women’s active participation in a peace process can make a significant difference." -- Nobel Laureate @LeymahRGbowee on the important role of women for peace.\xa0https://t.co/XHnRXGPlQk\xa0via @AfricaRenewal\xa0https://t.co/1XH5JbBegt\xa0Feb 01, 2020\xa0'
print(insertor.preprocess(tw_flat))
print(insertor.preprocess(tw_db))
print(insertor.preprocess(tw_flat) == insertor.preprocess(tw_db))

# %%
# Still problematic tweets from @DrTedros
dr_ha = [up[-1] for up in to_update]
df_dr = df[df["handle"] == "@DrTedros"]
df_prob_dr = df_dr[~df_dr["tweet_id"].isin(dr_ha)]

#%%
# Issue
# line 5851 of DrTedros_flat.jsonl
# hash 1526566693 tweet_id 1239591758520029185
# &amp; not stripped
# -> solved

tw_flat = 'RT @mugecevik: Dr @DrTedros "we have seen a rapid escalation in social distancing measures, like closing schools &amp; cancelling events &amp; othe…'
tw_db = 'RT @mugecevik: Dr @DrTedros "we have seen a rapid escalation in social distancing measures, like closing schools & cancelling events & othe…\xa0Mar 16, 2020\xa0'
print(insertor.preprocess(tw_flat))
print(insertor.preprocess(tw_db))
print(insertor.preprocess(tw_flat) == insertor.preprocess(tw_db))

#%%
# Issue
# line 6173 of DrTedros flat
# hash 1139880183 tweet_id 1234951668044894214
# "\xa0" (non-breaking space) was not stripped
# -> solved
tw_flat = "I and my daughter Blen really appreciate the lovely birthday wishes you sent us. Much gratitude! https://t.co/ww6voBnJfu"
tw_db = "I and my daughter Blen really appreciate the lovely birthday wishes you sent us. Much gratitude!\xa0https://t.co/ww6voBnJfu\xa0Mar 03, 2020\xa0"
print(insertor.preprocess(tw_flat))
print(insertor.preprocess(tw_db))
print(insertor.preprocess(tw_flat) == insertor.preprocess(tw_db))

# %%
# Still only 51/107 of DrTedros are found
# Issue
# hash 1449155825 tweet_id NA
# account "@toto_sparrow" deleted
# impossible to know exactly how many
# -> this tweet was manually deleted
tw_flat = None
tw_db = 'RT @toto_sparrow: Today is World Hearing Day. Remember that with a little extra attention, people with hearing impairment can be full parti…\xa0Mar 03, 2020\xa0'

#%%
# Issue
# line 6175 of DrTedros flat
# hash 1100886297 tweet_id 1234947288725966849
# for short tweets, the date "Mar03,2020" is still included in the comparison
# -> solved 58/107
tw_flat = 'Thanks so much my friend Toto. How thoughtful of you! https://t.co/rR08hxnv7q'
tw_db = 'Thanks so much my friend Toto. How thoughtful of you!\xa0https://t.co/rR08hxnv7q\xa0Mar 03, 2020\xa0'
print(insertor.preprocess(tw_flat))
print(insertor.preprocess(tw_db))
print(insertor.preprocess(tw_flat) == insertor.preprocess(tw_db))

#%%
# Issue
# line 6207 of DrTedros flat
# hash 1165309955 AND 1451999741 tweet_id 1234077816897097728
# can be multiple matches
# -> solved: notify and print the multiple ids but keep the last one found
# manually purge database of duplicates
# Duplicates are tweets with "theme_hardcoded" set to NULL instead of 0, must have been
# introduced then.
tw_flat = 'I am grateful for the contribution being announced @KSRelief_EN of $10.5 million to support @WHO to #EndMalaria in Yemen for almost 7 million people and the treatment of nearly 780,000 people over 18 months. #RIHF'
tw_db = 'I am grateful for the contribution being announced @KSRelief_EN of $10.5 million to support @WHO to #EndMalaria in Yemen for almost 7 million people and the treatment of nearly 780,000 people over 18 months. #RIHF\xa0Mar 01, 2020\xa0'

fake_db = [(
    "1165309955",
    0,
    "00/00/0000",
    "@DrTedros",
    "United Nations",
    None,
    tw_db,
    "0",
    "New",
    None,
    None,
    None,
    None,
    None,
    None,
    None,
),
(
    "1451999741",
    0,
    "00/00/0000",
    "@DrTedros",
    "United Nations",
    tw_db,
    None,
    "0",
    "New",
    None,
    None,
    None,
    None,
    None,
    None,
    None,
),]

print(insertor.preprocess(tw_flat))
print(insertor.preprocess(tw_db))
print(insertor.preprocess(tw_flat) == insertor.preprocess(tw_db))
print(insertor.check_in_db(tws_flat[6206], fake_db))

# %%
# Issue
# check @RY ...:

# %%
# Issue
# url field in db can also be Null (and not simply == "0")
# -> see "url_null.py"