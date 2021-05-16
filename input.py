import main

while True:
    code = input('>>> ')
    if code == 'exit': exit()
    result, error = main.run('<stdin>',code)
    
    if error: print(error.as_string())
    elif result: print(result)
