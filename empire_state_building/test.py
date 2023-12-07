a = {'template/1.png': [True, True, False, True, True, False, True, True, True, True, True, True],
     'template/2.jpg': [False, True, True, True, True, True, True, True, True, True, True, True],
     'template/3.jpg': [False, True, False, True, True, False, True, True, True, True, True, True],
     'template/4.jpg': [True, True, True, True, True, True, True, True, True, True, True, True],
     'template/5.jpg': [True, True, True, True, True, True, True, True, True, True, True, True],
     'template/6.png': [False, False, False, False, False, False, False, False, False, False, False, False],
     'template/7.png': [False, True, False, False, False, False, False, False, True, True, True, True],
     'template/8.jpg': [True, True, True, True, True, True, True, True, True, True, True, True]}
b = {'template/1.png': [True, True, True, False, True, True, True, True, True, True, True, True],
     'template/2.jpg': [True, True, True, True, False, False, True, True, True, True, True, True],
     'template/3.jpg': [False, True, True, True, True, True, True, True, True, True, True, True],
     'template/4.jpg': [True, True, True, True, True, True, True, True, True, True, True, True],
     'template/5.jpg': [True, True, True, True, True, True, True, True, True, True, True, True],
     'template/6.png': [False, True, True, False, False, False, True, False, False, False, False, False],
     'template/7.png': [False, True, False, False, False, False, False, True, True, True, True, True],
     'template/8.jpg': [True, True, True, True, True, True, True, True, True, True, True, True]}

if __name__ == '__main__':
    acnt = 0
    for k in a.keys():
        l = [1 for bb in a[k] if bb]
        acnt += len(l)

    bcnt = 0
    for k in b.keys():
        l = [1 for bb in b[k] if bb]
        bcnt += len(l)
    print(acnt, bcnt)
