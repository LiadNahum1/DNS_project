userIps = [("173.27.225.202",1),("173.27.225.197",2),("173.27.225.182",3),("173.27.225.164",4),("173.27.225.158",5),("173.27.225.153",6),("173.27.225.151",7),("173.27.225.144",8),("173.27.225.126",9),("173.27.225.118",10),("173.27.225.116",11),
           ("173.27.225.115",12),("173.27.225.113",13),("173.27.225.111",14),("173.27.225.102",15)]


def checkMyIp(x) :
    for ip in userIps :
        if x==ip[0] :
            return  ip[1]
    raise Exception("Ip not part of the organization") #Alert admin

def applyClassifier() :
    #TODO : apply the classifier on the chunck
    return 1


def checkUser(chunck,id):
    ident = applyClassifier(chunck)
    if ident != id :
        raise Exception("Alert admin")

def detector(x) :
    user1,user2,user3,user4,user5,user6,user7,user8,user9,user10,user11,user12,user13,user14,user15 = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    for line in x :
        curr_ip = x['ip.src']
        user = checkMyIp(curr_ip)
        if user == 1 :
            user1 = user1 + x
        if user == 2 :
            user2 = user2 + x
        if user == 3 :
            user3 = user3 + x
        if user == 4 :
            user4 = user4 + x
        if user == 5 :
            user5 = user5 + x
        if user == 6 :
            user6 = user6 + x
        if user == 7 :
            user7 = user7 + x
        if user == 8 :
            user8 = user8 + x
        if user == 9 :
            user9 = user9 + x
        if user == 10 :
            user10 = user10 + x
        if user == 11 :
            user11 = user11 + x
        if user == 12 :
            user12 = user12 + x
        if user == 13 :
            user13 = user13 + x
        if user == 14 :
            user14 = user14 + x
        if user == 15 :
            user15 = user15 + x
    checkUser(user1, 1)
    checkUser(user2, 2)
    checkUser(user3, 3)
    checkUser(user4, 4)
    checkUser(user5, 5)
    checkUser(user6, 6)
    checkUser(user7, 7)
    checkUser(user8, 8)
    checkUser(user9, 9)
    checkUser(user10, 10)
    checkUser(user11, 11)
    checkUser(user12, 12)
    checkUser(user13, 13)
    checkUser(user14, 14)
    checkUser(user15, 15)


if __name__ == '__main__':
    #create x example
    x = []
    detector(x)

