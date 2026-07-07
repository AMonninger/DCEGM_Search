from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType

agent = IndShockConsumerType()
agent.solve()

inc = agent.IncShkDstn[0]
trn = agent.TranShkDstn[0]
prm = agent.PermShkDstn[0]

print("IncShkDstn type:", type(inc))
print("IncShkDstn atoms:", inc.atoms)
print("IncShkDstn pmv:", inc.pmv)
print()
print("TranShkDstn type:", type(trn))
print("TranShkDstn atoms:", trn.atoms)
print("TranShkDstn pmv:", trn.pmv)
