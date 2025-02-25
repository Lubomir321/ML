import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def make_meshgrid(x, y, h=.02, lims=None):
    """
    Create a mesh of points to plot in.

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    if lims is None:
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
    else:
        x_min, x_max, y_min, y_max = lims
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, proba=False, **params):
    """
    Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    if proba:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, -1]
        Z = Z.reshape(xx.shape)
        out = ax.imshow(Z,
                        extent=(np.min(xx), np.max(xx), np.min(yy),
                                np.max(yy)),
                        origin='lower',
                        vmin=0,
                        vmax=1,
                        **params)
        ax.contour(xx, yy, Z, levels=[0.5])
    else:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_classifier(X,
                    y,
                    clf,
                    ax=None,
                    ticks=False,
                    proba=False,
                    lims=None):  # assumes classifier "clf" is already fit
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1, lims=lims)

    if ax is None:
        plt.figure()
        ax = plt.gca()
        show = True
    else:
        show = False

    cs = plot_contours(ax, clf, xx, yy, alpha=0.8, proba=proba)

    labels = np.unique(y)
    if len(labels) == 2:
        ax.scatter(X0[y == labels[0]],
                   X1[y == labels[0]],
                   s=60,
                   c='b',
                   marker='o',
                   edgecolors='k')
        ax.scatter(X0[y == labels[1]],
                   X1[y == labels[1]],
                   s=60,
                   c='r',
                   marker='^',
                   edgecolors='k')
    else:
        ax.scatter(X0, X1, c=y, s=50, edgecolors='k', linewidth=1)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    if ticks:
        ax.set_xticks(())
        ax.set_yticks(())

def main():
    X_train = np.array([[-0.2838956567996067, -0.9514577062421715], [0.5336243724331164, 1.2391807092747544], [-1.1733781500398293, -0.12125818785410411], [-1.1933837919166443, 0.7977471661585251], [-0.3749022010253586, -0.399647888966082], [-1.1167888015212781, -0.36305797619691144], [0.039256792917955204, -1.0394931130341973], [-0.4604969015467236, 0.7922061500363433], [0.37337223417648785, 0.4494555883781123], [-0.568303256494455, -1.1232602350503744], [-0.81082537661138, -0.6711970101754818], [-0.22345689478621258, 0.5013000632757527], [0.3313951279985257, -0.8810874882274999], [1.2318792490706367, 0.6363072629422297], [-0.11943856553240402, -1.7003144998656494], [-0.35643238131183475, 0.8175979474973925], [0.3683968441602589, -0.48940507796671934], [-0.25537363880473685, -0.41807533266251845], [-1.326453469243338, 1.0434498669985466], [-0.5928504742384184, -0.8014219967259865], [0.7279619024896092, -1.124898213385803], [-2.135114806556929, 1.5325663378829528], [0.23054207298658314, -0.8688038030980288], [1.09813977127213, 0.5834095372606052], [0.8383365301398581, 0.6171095674617604], [0.4864382365742221, 0.3066693831497788], [0.7869154395775334, 0.44466709694755024], [-0.4789325787710836, -0.5665398506165835], [-0.2033005538076183, 0.936879326793838], [0.33916583122064325, 0.8125340115773076], [-0.36173327639426744, 0.3635405492005922], [1.186033630229497, -2.0135971555205883], [-0.4872537569255583, -0.6374460814453928], [-0.1886984966469057, -0.582874620674272], [0.9814116207131341, 1.339417381990318], [1.501892989129474, 1.0308120604990516], [-0.06444265361075792, -0.3811263385729394], [1.2883973986126178, 1.54321683128406], [-0.9023792536753112, 0.21291292489083982], [-0.2759437902492382, 0.6621904099788878], [0.0001858754887992525, -2.226878517445455], [-0.1362184855982857, -0.6121435819533576], [-0.8011893090956812, -0.7315716015582898], [-1.3111436026329002, -0.1111601072948019], [-0.18271588215743786, 0.3457423406880571], [1.6257798371350645, 0.0982365559249859], [1.1875422054825977, -0.9880472562504043], [0.33089634796715883, 1.741470811075398], [1.0631946323865333, 0.7257074768137395], [-0.06311727766789436, 0.5031383381776916], [0.37022163866922086, 1.2731536600997202], [-0.003144439890306582, -0.2643845828077005], [0.35748430080225196, -0.02276577723535955], [-0.7108893917705611, 1.546883578777394], [0.7623299726607934, 1.332544052760334], [-1.5523355349329786, -1.523374405415224], [-1.0033968871019354, 2.1186213983936404], [0.7228231929278502, 0.3354057061774236], [0.09127552532312139, -0.5044686411488244], [0.049511493424470455, -0.24469010641021963], [-1.4030996802581286, -0.5123954158147642], [-0.7755016896935467, -0.6023941140640741], [1.2443537802432727, 1.0531488755822307], [-0.022313084754849173, 1.8740327882287748], [-2.227798161139633, 1.2704686733325055], [0.04429289902471703, 0.5975494463699031], [1.140986130566655, 0.8514906303904557], [1.4416504559593393, -0.8666002086070336], [0.13229513967793535, 1.6265341302036918], [0.036930844122167815, 1.6778645263139464], [-0.6129759829733853, -1.3540585025541816], [0.822080441136708, -0.7454482265336125], [-1.0301545489709054, 1.5782845063679125], [-1.273928252534737, -0.11984875704626896], [-0.4228825084943045, -0.30596176886428644], [1.0512737826798484, -2.2905228855219195], [-0.7895545008204998, 0.0008570371683366541], [0.373018739503365, -0.49846837322095716], [-0.7215506771427015, -0.5228212713894625], [1.9102170734661015, -1.527008694928588], [0.6186714540108945, 0.5312702334935516], [-1.1210455850133891, 2.876880268219743], [1.2106174087990125, -0.5486163698067474], [-1.2568061642025437, -0.06132817062350283], [-1.507567058928723, 0.4183407652970397], [0.7111331229704009, -0.7151138952984033], [-0.42262009248250265, 0.5792871662469677], [-0.934751977714784, 0.03603874869905847], [1.2113565037931928, -0.578726751548018], [-1.1532017104078844, -0.04859690867590418], [2.621717414728499, 0.7010104269714286], [-0.014867328444009172, -0.9339672860451339], [-0.47119368546592727, 0.14829702820374538], [2.1974859408744485, 0.08185797863724965], [0.8953265394719347, -0.8203589034678554], [-0.6171110796414327, 1.2757126416282951], [1.0994833486640467, -1.260336339627403], [-0.2596021941989571, 0.762758510234427], [0.8268613568891915, -0.8985244518298044], [1.173280596835973, -0.20297282877826645], [0.09446526747965596, 1.112636393566609], [0.07936850361273007, 0.29170741953920976], [0.8551349827856474, -1.8682026878835596], [-0.033337759206480404, -0.8214716831679265], [1.0819580117712548, -0.9127422506836348], [-0.9591149914122687, -0.9328836688032787], [-0.1099440581486897, 0.42916834100261914], [0.8475005569770568, 0.20054117830458326], [0.8979693008871347, -1.3447509649606335], [-0.7455492647781794, 3.38558393347755], [-0.23516794071610156, 0.11697171743005041], [0.38222884668943113, -1.8909819093765479], [0.44654831479661933, -0.19055020505140166], [1.0312540249497077, -0.6101758935097945], [0.8985336358172423, -0.8271916279779142], [-1.3355520473995055, -0.8751842227539924], [-0.38273042170978366, 1.3004446221391233], [0.07037710216523475, 0.36332442893417977], [-0.3825477774706774, -0.013343388431317568], [-0.9288311144977259, -0.12459521677753607], [-0.4180204187540841, 0.0448693923341529], [-1.1429759426033679, -0.42007662751448516], [-1.3408513985763257, -1.0469378848074313], [-0.575321695412197, -1.8732435437801316], [1.0517931429039158, 0.15069874357105625], [-0.24116815604217226, -0.436437276384716], [1.4692986003852984, -0.9036660029462869], [0.23942263961903051, -0.29011381171426326], [0.40850544789781107, 0.42771610946643224], [1.0308043260973696, -0.001549134042422927], [-1.0267671061808958, 0.7211554069168448], [1.2429876011664052, 0.6053524916935712], [0.09981587038836585, 0.7858824560030081], [-0.33098927346522805, 0.1879239194774707], [0.14642719541829644, -0.7749873858752286], [0.627556538917701, 2.1699232195908644], [0.19113387134761922, 1.7091843266890132], [-0.3870556704158065, -1.2362160276153784], [1.3046669699360534, 0.18911656601498777], [-1.0066411320971715, -1.820696884404112], [0.20404855366619476, 1.1870734683031552], [-1.3090715189818964, 0.05660347774004562], [0.3400834444297477, -0.9902223321976724], [0.21357317797805486, -2.9403707589876817], [-0.023608494443350024, -2.2783245456456593], [0.922675071286956, 2.141986582038585], [-0.885698363724953, -1.8238125488737222], [-1.5580195902516376, 0.758114659713307], [1.481916690593938, -0.7292290561554596], [3.458731515280756, 0.3450137324664156], [-0.3870805175929403, 1.116698882052221], [0.43876300065401863, 0.6125958123095384], [0.6729116404218808, 1.1442042871385674], [0.8936322812833307, 0.35464157203544566], [-0.6908074938255118, -0.4596348378810585], [-0.4157531078899093, 2.0783491067941195], [0.4177397482430752, 1.15217232977657], [-0.6495362184340839, -1.2977899214306658], [0.8799361589089428, -1.1703742011984002], [1.3296350902041216, 0.7778199697227365], [1.675768484724197, 0.37197812861091906], [-1.0080826024988505, 0.25784357217122594], [1.5301538421039664, 0.22764384031338913], [-0.6187543180403328, -0.6419546167397391], [-0.8711074520666537, 0.311477892143458], [0.6376229871027373, 2.0314669142844206], [-0.3109805264019763, -0.17903668975726933], [-0.6454682837566055, -1.2549101478265527], [0.4711589471013035, -2.396410537765022], [-0.6470384174569979, 0.7457400488393573], [0.11189335527013111, 0.09238204007586484], [0.24622647768169628, 0.3130629063928071], [-0.10108903911697228, 0.03628493787480477], [0.17165956777431848, -1.3592763022207948], [0.3404809598717748, 1.1667483182113867], [-0.2566184769972755, 2.081527219322955], [1.533155148307294, 0.2887281227476182], [-0.5314555904390706, 0.4131387291924458], [1.395596337189551, 0.12111065289488156], [-0.22147247859422184, 0.5406422916668505], [-0.4962508608339872, -0.3333202999762297], [0.6164390643330384, -1.0354233836909514], [-3.12897612735514, 0.4763620130725258], [-0.6994377888086141, -0.17583072421481505], [1.7151958659114874, 0.10160263790710036], [0.5191860622816569, -1.3549217333228278], [0.6822045962850344, 0.8734339676873084], [-1.0858158534034055, 0.1751681660967047], [-0.42290728166924846, -0.5651070609478825], [-0.3715559285587616, -1.423057137093781], [-0.8215770032778047, 0.5280104946240772], [0.07868700713013004, 0.09248435502042102], [-0.21619311944551142, 0.5861933805501323], [-1.2254835891291116, 0.24884680150140404], [-1.5500006418405843, 1.565189802107864], [-0.960286755287913, -0.340835920215301], [-1.2142011839328535, -0.35418342067839126], [-0.6179414765315178, 0.0849677284067546], [-1.268216972841844, 1.5363450332495612], [-0.10412151100427762, -0.30934558976037624], [-0.3240893134289809, 0.09146845014357063], [-0.3460517311988248, 0.27194841188570557], [0.5491680515413014, -0.07033005338666115], [-0.06295546986759167, -0.08281040115586177], [-0.16655290317877358, -0.681026187021237], [-1.865191534590544, -0.8101622921482667], [0.05665935368075517, -0.41874585393505387], [2.347110211621567, -0.032788292655735476], [0.6423449932971251, 0.4656354527653333], [0.8195809832345181, -1.3803483214545405], [-1.2752561455447513, 1.6259240367550731], [0.41899062229085876, -0.23421781855284762], [-0.6454992175210916, -0.16806827093836774], [0.5871156852106701, 1.269680480125934], [-1.3899442121269892, -0.317576786656503], [1.3327979343016993, 0.5911254416400183], [-1.1498934205700113, -0.6181271747112427], [-0.4786907150572737, -1.6045011473530222], [1.5454150653176713, 0.1552760175939987], [0.7284239867349964, -0.33038015505285784], [-1.0789955687554862, 0.9567531372284767], [1.731585357960964, -1.4593076774617855], [-0.8508591907975425, 0.9467011401431427], [0.6851899322020121, -0.5030687646990921], [0.667027648403215, 0.6861371040463322], [2.520374775157467, 0.6254138971627148], [1.2337200488971198, 0.2516032835010752], [0.05811840532383392, 0.045113724461615365], [-0.20177528778231713, -0.8084146687271709], [-2.8324146504748127, 0.7235695241199461], [1.1179434047365815, 0.26616942616764266], [-1.4841452641834496, 1.418024035992439], [-0.21799755710821953, -0.42703127321793083], [1.051311698616165, 1.03823987403497], [0.47347858882510324, 2.228508434587535], [0.3790731500482539, -0.40432058822395683], [-0.4852488696803753, -0.4634226781093403], [-0.6525132918376494, -0.08006674061217081], [-0.11063899421272934, -0.09528937443994864], [0.9722781494365388, -0.6816958168977075], [-0.5747044247311796, 0.5467437614308968], [0.26121193652293206, -0.09005179803430244], [-0.31376660772582665, 0.29985874162822224], [-0.753824273769213, -0.3620561625210939], [-0.8575376888272193, 2.7099197953941845], [0.11761285960897234, 0.8068249764629332], [0.5636971087320481, 0.6456961115418114], [-1.4360062646626377, 0.5594754654150109], [-0.6518041044331592, -0.6721138682381648], [0.7442534545418812, 0.5707177604518417], [-0.5963437602308062, -1.1896425008125247], [0.5965002580271624, 2.0051348002176272], [0.9463395455084941, 0.825057695645937], [-0.9495522551896719, -2.127077181377807], [0.7777843279895834, 0.4127403593520303], [-0.02148723393669681, -0.32188153909992345], [0.46414233679566186, 0.4712651707146057], [-1.0016225386609152, 0.055950440434399076], [0.422650851672825, 0.7078348317989733], [-0.4190919260545995, 0.0004214465049275508], [0.3805321359698532, -0.8624968258404396], [-0.6880951926282698, 0.003837240139904667], [-1.3829673696264209, 1.0177310238723354], [-0.15672761369967217, -0.10062591716632141], [-0.7027905585217172, -0.5342644816895001], [0.4536742283411869, 0.22639972280111365], [-0.3709770240354775, -0.9025326465740529], [0.9957398690555221, -0.42982377178482123], [0.32163304679415516, -0.4819967366319121], [-0.35976107340706787, -0.5194866604140819], [0.26394916894724235, 2.545470951998665], [0.4002024101786496, 2.1615903579346556], [0.854602127492853, -1.0854472911049289], [0.1174096465668036, 0.006721648864677999], [-2.9691626878701856, -0.7242989997451593], [0.8987341643254558, 2.033601054229681], [0.9148851994832734, 2.0879572720401014], [0.3522259503009353, 0.11180739629866294], [-0.502642449057185, -0.8948815109563395], [-0.20792603091017914, -0.22600784415524908], [-0.29806749478079914, 1.2100797210184582], [0.5595527100188603, 0.6168569243736542], [-0.21333132784597753, -1.3717690716819522], [-0.7715922979434555, 0.2565228534044801], [-0.0891009324080554, 0.585928029218626], [-0.6591580601906178, 0.8931401838714205], [-0.16625652933376764, -0.9800959359963065], [-1.4529363408644527, -0.679306037862509], [0.2584595333877436, -1.5950770144298727], [-0.61656264198897, 0.33755386850300917], [1.9933237983716887, 0.3761006760505031], [1.3174990375153608, 1.1452537010927148], [0.677627651940823, -2.6902841238644988], [-0.724054640964135, -0.3000288281338989], [-0.2159082497588626, 1.7336102606421795], [-0.4266279025151314, 0.03950053448675032], [1.1347373226482904, -0.2064684772359824], [1.5681580322398938, -0.1474555080076672], [0.7867151203589299, -0.2320628756245722], [0.9879331662305725, 0.16084361789779578], [1.9072916263299236, 0.12740931245402892], [0.3593558532824468, -1.0464082084805657], [-0.9466471820231397, 0.6220988253723009], [-0.24668927972799742, 0.25730727112215], [0.2678004513766659, 0.44099892391696216], [-1.8337168425892258, -1.0763543254644614], [-0.07323187627335354, 1.2598582295590424], [0.6972743400919953, -0.4562013612702838], [0.09154699742303662, 0.3472798725846255], [-0.1342154953792364, 1.5363400607981663], [-0.20195958932763897, 0.7400561560254206], [1.2953621117573633, -1.791852315341229], [0.6359874506550378, 0.013472425825908131], [-0.07521860566579358, 0.23704890901630576], [-0.7388588074124536, 0.004310695150183216], [2.764857830038035, -1.0453281311215454], [-0.8282864085458365, 0.49613903081906535], [0.672941893299038, -0.3082562042178742], [0.35632095550459814, -0.17185420969816534], [-0.18305245208793622, 0.9937786733873119], [0.60716232436464, -0.4879400093850765], [-1.4572470824976023, -0.5572880150506218], [-0.01957132870560036, -0.7270610607628591], [-0.3035958073004957, 0.2453103262654795], [0.4257480843036171, -0.27304496380455257], [1.2804292495843579, -2.1726569843484675], [1.1361628367058219, -0.7017506468024972], [-0.5156204432182062, 0.42742986291911966], [1.1756801696479728, -0.10993635902802859], [-1.3641184675495504, -0.03620753608628131], [-0.30333878876868275, 2.3251644991277307], [0.4212651989111933, -1.4818460761850987], [-1.563566639941443, 0.2439798288036792], [-1.8937009445533248, -1.3485785094916285], [-1.8136967328699336, -1.0403594315768732], [-1.3048701632957955, -0.06097077990690114], [-0.38735022866483515, -1.2015676660525603], [1.851166182207869, -0.024776528689089503], [0.2556091313903498, 0.4438176155353519], [-0.7234229995855097, 0.6879366499302098], [0.7196758530024778, 1.638075385003611], [0.4075285862643082, -0.4578759583126973], [1.2723915976205977, 0.7541976114828907], [-0.7330848109785708, -1.2262587008379926], [-1.0014668632511217, 0.8225870110487233], [0.8742803059215956, -0.042808754216316316], [-2.602540752953433, -0.10437429522583803], [0.24712965292818195, -0.2393470952320842], [-1.3219253144537169, -0.24700283886514782], [-1.503427723042993, 0.8142053207665486], [1.3189469269741019, -0.7793053766659751], [1.0711485577431354, 0.5619955512319068], [-0.9480560157891544, 0.4628841419682679], [-1.47049628681759, -0.31081039520445675], [-1.5224611120477884, -1.5008310465762378], [0.6623943556175192, 0.23276206825805015], [-1.776469535406049, 1.8079691184519386], [-1.339195756729503, 0.4008429001122398], [0.4550712078261438, -0.8875712982995663], [-0.009270202446999965, -1.0267846839451114], [0.7877151487140844, -0.47496542300055067], [-0.6802875136577102, 0.7018277856150601], [0.23466416033260967, 1.0268342448510321], [0.008365868847567837, -0.20558112176789362], [1.875477361878918, 0.8758699033958063], [-0.34795503747550877, 1.9529068839264725], [-0.7670389885635669, 0.09109601206860687], [0.26303375291433445, 0.47297031827294433], [0.03099437119739402, 1.953941334036071], [-1.134331095909403, -1.9540817418900671], [-0.09016317243900761, -0.3386373556202356], [0.11010331723090928, 1.6121697784673377], [1.4996113260162651, 0.2217299679842536], [-1.1824359517421954, 0.9178431662952057], [-1.1543867482806869, -1.1945195933654447], [-1.9690837184583287, 1.6176885497077529], [1.0456735515552804, 1.0787534619047214], [0.7727053010631211, 0.99882374915609], [-1.359976516724723, -0.4987688954307049], [-0.9048345838665076, 0.2339219856085942], [-0.4951780155050511, -0.04391952446593599], [-0.5939388492475773, 1.4282025386294623], [0.5593040941211391, -1.158738622876123], [0.2160405278864936, -0.08173800878667414], [0.19152800130366449, -1.6932592525204544], [1.1043416970032802, 0.2843403393189035], [-1.095805125100648, 0.5868980985216432], [0.6323284973606275, -0.5913753114719108], [2.0361413215812973, -0.390471671917144], [0.5388162109592907, 1.2591417383808787], [1.037043470507998, -0.2236067084574582], [1.1937276578584535, -1.1421018288847962], [1.513551337899092, -0.5839711825238769], [-1.0718251218498442, -0.6526050828920889], [-0.047721439943374445, 0.5161682298470723], [-0.7411558154320627, -0.2031273003437558], [-0.5262192093973934, 0.9803011553966003], [-1.7274190858436376, -0.16562173304395084], [0.6085615235366635, -0.1453705105704001], [0.2684176088766702, 0.4104236304192945], [3.356391183390804, -0.1763971758731256], [0.7242579805864809, -0.3603393819316337], [0.10687528571970586, -0.18546144818027221], [-2.1754766177027727, 1.315723162873682], [0.7059061317852813, -0.0021652999521815887], [-1.3769237557654614, -2.090795391398622], [0.24622994242326215, -1.1274433410403666], [0.8025237440774234, 0.07754913942170383], [1.9170675843841245, 1.4318341399770698], [0.3732420869781553, -2.2722792734181976], [-1.4416129436177132, -0.9971088721248295], [-0.5288574500306227, -1.8742474681029], [1.4325873599042147, -0.03371130814087855], [0.20723465531700058, -0.3090768704418818], [0.39021509914823715, 0.8720541711310863], [0.7451619181508513, -1.8741784778768733], [-0.8992878044767934, 0.40273048879844037], [-0.49138947738406324, 0.3076857668380193], [-0.5119386557903507, 1.4758537090826036], [1.5230060220952923, 0.25851475741114277], [0.0424973478185026, 0.25746136094391325], [-0.9040751152760257, 0.9964218585474195], [-0.2101088872496639, 0.044470877810113996], [1.0367662060952099, -0.8929491230266268], [0.4534759537749111, -0.08413712820605906], [1.2706178855710795, -0.1678404584679596], [-0.2800654645951509, -2.0321146794901517], [-2.3081391205527306, -1.2930495755193434], [-0.7743561880161488, 0.3195740127999761], [0.18788731245868562, -0.24536295665565905], [-0.47293705276352827, -1.2760064533197566], [1.2227224437623232, 0.2879618121419084], [-1.4793816831410829, 0.4782333858937716], [0.34408977798996127, 0.9243695690128633], [-0.14782033560405947, 0.17883625309353793], [-2.6131308570929477, 0.7062821235166662], [2.0603541789930895, 1.2124343599863712], [-0.9605490641490254, -2.2924325978962017], [0.4330271713327844, 1.1686869398470365], [0.9036173200548645, -0.9074289119932806], [-0.4293497021910093, 0.20868209759377498], [1.187237806140174, 0.9602224942708286], [0.8702320333772587, -0.7318009540276827], [-0.839153309578388, 0.42578027621313536], [0.9105854323540942, 0.483166021898615], [-0.14295517827548176, -0.34892691790874664], [-1.7610941758277547, -0.5476847328194702], [0.7129155654044202, -1.2051206372860541], [1.9986547070501, 1.1110702507645867], [0.6083981729048806, 0.06444237166315543], [-0.3156084083777615, 0.21463110339431166], [-1.05242232314278, -0.32902113165550523], [0.17153514333597672, 1.5832263535072475], [-0.6357730503018189, -1.5680362559066163], [-0.8898296692079306, -1.3974671285608056], [-0.5075868828998603, 1.222016424489662], [-1.6171193630938714, -1.4704843420640805], [-0.09297687101652785, -1.7526288424754444], [0.36913868957042023, 0.8058579121882057], [0.29467177354251883, 0.5809446130862611], [0.3709369825114636, -0.09857102050348358], [-0.9257080808850393, -0.7228942799075551], [-1.1154883092546273, -0.9576943173898841], [-1.1866976253097048, -0.7424970451522646], [-0.2248407542265962, -0.059305906384528734], [-0.6416675856684165, 0.5372840988283111], [0.2373965319806633, -0.5767000639714456], [2.5427255022269866, -1.6305808823204098], [-0.8873731813648268, 0.019520256770537824], [0.3848476069215733, -0.7510084169265645], [-1.4628297779915633, 0.3081998264580409], [-0.3667023164044164, -2.2358230518770186], [0.6436644494718863, -0.8326844078848099], [-2.0418321811395783, 0.48779518262939764], [1.135156248442037, -0.9854091891628719], [1.2841050458880063, 1.438814046491594], [0.5859792673523605, 0.7466022147175375], [-0.6933762094513539, -0.8443767396817398], [1.4545191603995604, 0.10547631172033595], [0.23860933447757757, -0.04315435995384592], [-0.5128638516664976, -1.355501936777571], [-1.2447216861382036, 0.1624951168250283], [-0.21272068790627513, 0.40686102291904], [0.3635301102291655, 1.2593833804495187], [1.1005810932289657, 0.9847061226000355], [-0.5357406822177201, -1.74105144404998], [0.5357148405468785, 0.5416930196108234], [-0.9704681220415713, 1.1489081816347628], [-1.0998893585740857, -0.47805239558315654], [1.2792040614042974, -0.808848980671474], [-1.367984240876195, 0.39985505654802356], [1.0106561840523807, -0.2623145577297814], [1.3457618683984949, -0.4472512841752605], [-0.368049070110498, -0.3434373891923286], [-0.496020550102438, -0.462734100263541], [-1.240621489754672, -2.1363880521809575], [-0.5525478868555893, -1.1035107801398463], [0.885081661274038, 0.4383627301840887], [0.9438837600072746, 0.012622706535779665], [-0.042991704695849495, 0.7785201479872178]])
    y_train = np.array([4, 2, 4, 1, 4, 4, 4, 1, 2, 4, 4, 4, 4, 2, 4, 2, 4, 4, 1, 4, 3, 1, 4, 2, 2, 2, 2, 4, 2, 2, 4, 3, 4, 4, 2, 2, 4, 2, 4, 2, 4, 4, 4, 4, 4, 3, 3, 2, 2, 2, 2, 4, 4, 5, 2, 4, 5, 2, 4, 4, 4, 4, 2, 2, 1, 2, 2, 3, 2, 2, 4, 3, 5, 4, 4, 3, 4, 4, 4, 3, 2, 5, 3, 4, 4, 3, 4, 4, 3, 4, 3, 4, 4, 3, 3, 1, 3, 2, 3, 3, 2, 2, 3, 4, 3, 4, 2, 2, 3, 5, 4, 3, 4, 3, 3, 4, 2, 2, 4, 4, 4, 4, 4, 4, 2, 4, 3, 4, 2, 3, 4, 2, 2, 4, 4, 2, 2, 4, 3, 4, 2, 4, 4, 3, 4, 2, 4, 4, 3, 3, 2, 2, 2, 2, 4, 5, 2, 4, 3, 2, 3, 4, 3, 4, 4, 2, 4, 4, 3, 1, 4, 2, 4, 4, 2, 5, 3, 4, 3, 2, 4, 3, 4, 4, 3, 3, 2, 4, 4, 4, 4, 4, 2, 4, 1, 4, 4, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 3, 2, 3, 5, 4, 4, 2, 4, 2, 4, 4, 3, 3, 1, 3, 1, 3, 2, 3, 2, 4, 4, 4, 2, 1, 4, 2, 2, 4, 4, 4, 4, 3, 4, 4, 4, 4, 5, 2, 2, 4, 4, 2, 4, 2, 2, 4, 2, 4, 2, 4, 2, 4, 4, 4, 1, 4, 4, 2, 4, 3, 4, 4, 2, 2, 3, 4, 4, 2, 2, 2, 4, 4, 2, 2, 4, 4, 2, 1, 4, 4, 4, 4, 3, 2, 3, 4, 2, 4, 3, 3, 3, 2, 3, 4, 4, 4, 2, 4, 2, 3, 2, 2, 2, 3, 2, 4, 4, 3, 4, 3, 4, 2, 3, 4, 4, 4, 4, 3, 3, 4, 3, 4, 5, 4, 4, 4, 4, 4, 4, 3, 2, 4, 2, 4, 2, 4, 1, 2, 4, 4, 4, 4, 3, 2, 4, 4, 4, 2, 5, 4, 4, 4, 3, 4, 2, 4, 2, 5, 4, 2, 2, 4, 4, 2, 3, 1, 4, 1, 2, 2, 4, 4, 4, 5, 3, 4, 4, 2, 4, 3, 3, 2, 3, 3, 3, 4, 2, 4, 1, 4, 2, 2, 3, 3, 4, 1, 2, 4, 4, 2, 2, 3, 4, 4, 3, 4, 2, 3, 4, 4, 5, 3, 4, 1, 4, 3, 2, 3, 4, 4, 4, 4, 4, 2, 4, 2, 4, 4, 2, 4, 2, 3, 4, 2, 3, 4, 2, 4, 4, 3, 2, 2, 4, 4, 2, 4, 4, 1, 4, 4, 2, 2, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 3, 4, 3, 2, 2, 4, 3, 4, 4, 4, 4, 2, 2, 4, 2, 1, 4, 3, 4, 3, 3, 4, 4, 4, 4, 2, 2, 2])


    ovr_model = LogisticRegression(multi_class='ovr', C=100, random_state=42)
    ovr_model.fit(X_train, y_train)

    multinomial_model = LogisticRegression(multi_class='multinomial', C=100, random_state=42)
    multinomial_model.fit(X_train, y_train)

    binary_model = LogisticRegression(C=100, random_state=42)
    binary_model.fit(X_train, y_train==1)



    plot_classifier(X_train, y_train, multinomial_model)
    plt.title("lr_mn(one-vs-rest)")
    plt.tight_layout()
    plt.show()
    plot_classifier(X_train, y_train, ovr_model)
    plt.title("lr_ovr(one-vs-rest)")
    plt.tight_layout()
    plt.show()
    plot_classifier(X_train, y_train, binary_model)
    plt.title("lr_class_1(one-vs-rest)")
    plt.tight_layout()
    plt.show()

    #1 The graph shows well separated dots
    #2 Some classes fall into other areas
    #3 Unable to differentiate between the green and purple class
if __name__ == '__main__':
    main()