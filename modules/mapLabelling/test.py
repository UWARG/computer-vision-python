#print ("Hello World")

list1 = [[43.47036890733803,-80.5362815379075],
        [43.471065,-80.537680],
        [43.470699,-80.537885],
        [43.470794,-80.538194],
        [43.470670,-80.538825],
        [43.47073421526762,-80.53905567172941],
        [43.47119262569689,-80.53951567055144],
        [43.47108945932336,-80.53988045097043],
        [43.47130747107536,-80.54003065466938],
        [43.471437888450296,-80.54010307427907],
        [43.471584364346576,-80.53968330859445],
        [43.47243693457104,-80.54091980691322],
        [43.47289825180128,-80.53985496997458],
        [43.47276394461899,-80.53975036380857],
        [43.472178050369685,-80.53954115149237],
        [43.47197366731934,-80.53935071465979],
        [43.47202622302512,-80.53908785816459],
        [43.471235936072276,-80.53785940644642],
        [43.47162718779775,-80.53752413031066],
        [43.47202233000695,-80.53672483204367],
        [43.47220530138471,-80.5368133449377]]

with open('write.txt', 'w') as f:
    f.write('\n'.join([','.join(['{:4}'.format(item) for item in row]) 
        for row in list1]))