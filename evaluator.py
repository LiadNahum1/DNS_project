userIps = [("",1),("",2),("",3),("173.27.225.164",4),("173.27.225.158",5),("",6),("",7),("",8),("",9),("",10),("",11),
           ("",12),("",13),("",14),("",15)]
dnsIps = ["172.31.0.2"]

def checkDnsServerIp(x):
    if x in dnsIps :
        return 1
    return 0

def checkMyIp(x) :
    if x in userIps :
        return 1
    return 0

#def detector(x) :
    # x - 30 min of user request
    # check if all requests from supposed ip
    # check if all requests got to dns server in org
    # apply network data analyser
    # apply requests (name,time) analyser
    # if pass threshold accept the chunck
