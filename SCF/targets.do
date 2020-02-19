log using tagets.smcl, replace

// a. load data

use SCF_04_small.dta, clear
// taken from official replication material: 
// http://benjaminmoll.com/wp-content/uploads/2019/07/HANK_replication.zip

// b. Table C.1
sum housepos durables liqpos gb cb indirect_eq direct_eq netbus houseneg nrevccredit revccredit [fw=fwgt], d

// b. Table 5 (using footnote 37)
sum netbrliq netbrilliq income [fw=fwgt]

local cutoff 1000
gen poor_htm = (netbrliq > -`cutoff') & (netbrliq < `cutoff') & (netbrilliq <= 0)
gen wealthy_htm = (netbrliq > -`cutoff') & (netbrliq < `cutoff') & (netbrilliq > 0)
gen borrowers = netbrliq < -`cutoff'
sum poor_htm wealthy_htm borrowers [fw=fwgt]

tab borrowers [fw=fwgt], summarize(networth)

// c. Table 5: gini
inequal7 netbrliq netbrilliq networth [fw=fwgt]

// d. Table 5: top and bottom shares
pshare estimate netbrliq netbrilliq networth [fw=fwgt], percentiles(25 50 90 99 99.9) 

log close
