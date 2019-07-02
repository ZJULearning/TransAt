import ConfigParser

def process_config(conf_file):
  """process configure file to generate TrainSetParams, TestSetParams, NetParams, SolverParams 
  Args:
    conf_file: configure file path 
  Returns:
    TrainSetParams, TestSetParams, NetParams, SolverParams
  """
  trainset_params = {}
  testset_params = {}
  net_params = {}
  solver_params = {}

  #configure_parser
  config = ConfigParser.ConfigParser()
  config.read(conf_file)

  #sections and options
  for section in config.sections():
    #construct trainset_params
    if section == 'TrainSet':
      for option in config.options(section):
        trainset_params[option] = config.get(section, option)
    #construct testset_params
    if section == 'TestSet':
      for option in config.options(section):
        testset_params[option] = config.get(section, option)
    #construct net_params
    if section == 'Net':
      for option in config.options(section):
        net_params[option] = config.get(section, option)
    #construct solver_params
    if section == 'Solver':
      for option in config.options(section):
        solver_params[option] = config.get(section, option)

  return trainset_params, testset_params, net_params, solver_params
