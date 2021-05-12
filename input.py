import main

while True:
    code = input('>>> ')
    if code == 'exit': exit()
    result, error = main.run('<stdin>',code)
    
    if error: print(error.as_string())
    else: print(result)
