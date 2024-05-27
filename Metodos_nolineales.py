import sympy as sp
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

def graf(errors):
    iterations = list(range(len(errors)))
    fig, ax = plt.subplots()
    ax.plot(iterations, errors, marker='o', linestyle='-', color='r', label='Errores')
    ax.set_xlabel('Iteración')
    ax.set_ylabel('Error')
    ax.set_title('Gráfica de Errores')
    ax.legend()

    # Guardar la figura en un buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode()

    # Limpiar el buffer
    buf.close()

    return img_str

def biseccion(equation, a, b, tol, error_type):
    x = sp.symbols('x')
    relative_error = error_type == 'relativo'
    
    try:
        fx = sp.sympify(equation)
    except sp.SympifyError:
        return "La ecuación ingresada no es válida.", [], None

    iteracion = 0
    table = []
    errors = []

    fa = fx.subs(x, a)
    fb = fx.subs(x, b)
    if fa == 0:
        sol = str(a) + ' es una raíz'
        return sol, table, graf(errors)
    elif fb == 0:
        sol = str(b) + ' es una raíz'
        return sol, table, graf(errors)
    else:
        xm = (a + b) / 2
        fm = fx.subs(x, xm)
        e = abs(xm - a)
        if xm == 0:
            return "division por cero detectada", table, None
        if relative_error:
            if xm == 0:
                return "division por cero detectada", table, None
            e = e / abs(xm)
        row = f"it: {iteracion} |   xi: {a}  |   xm: {xm}  |   xf: {b}  |   e: {e}"
        table.append(row)
        errors.append(e)
        iteracion += 1

        while e >= tol and fm != 0:
            if fa * fm <= 0:
                b = xm
                fb = fm
            else:
                a = xm
                fa = fm
            xm = (a + b) / 2
            fm = fx.subs(x, xm)
            e = abs(xm - a)
            if xm == 0:
                return "division por cero detectada", table, None
            if relative_error:
                e = e / abs(xm)
            row = f"it: {iteracion}    |   xi: {a}  |   xm: {xm}  |   xf: {b}  |   e: {e}"
            table.append(row)
            errors.append(e)
            iteracion += 1

        graph_img = graf(errors)
        if fm == 0:
            sol = f'{xm} es una raíz'
            return sol, table, graph_img
        else:
            sol = f'{xm} es una raíz con tolerancia de {e}'
            return sol, table, graph_img

def busquedas_incrementales(equation, x0, max_ite, deltax, error_type):
    x = sp.symbols('x')
    try:
        fx = sp.sympify(equation)
    except sp.SympifyError:
        return "La ecuación ingresada no es válida.",[], None

    xf = 0
    fa = 0
    fb = 0
    it = 1
    evaluations = []
    fa = fx.subs(x, x0)
    evaluations.append("it: " + str(0) + "    |    x0: " + str(x0) + "      |       fa: "  + str(fa))

    if fa == 0:
        return str(x0) + " es una raiz", evaluations, None
    else:
        xf = x0 + deltax
        fb = fx.subs(x, xf)
        evaluations.append("it: " + str(1) + "    |    x0: " + str(xf) + "      |       fb: "  + str(fb))
        
        while it < max_ite and fa * fb > 0:
            fa = fb
            xf += deltax
            fb = fx.subs(x, xf)
            row = "it: " + str(it + 1) + "    |    xf: " + str(xf) + "      |       fb: "  + str(fb)
            evaluations.append(row)
            it += 1
        
        if fa * fb <= 0:
            if fb == 0:
                sol = str(xf) + " es una raiz"
                return sol, evaluations, None
            else:
                sol = "Entre " + str(xf - deltax) + " y " + str(xf) + " hay una raiz"
                return sol, evaluations, None
        else:
            sol = "Se superó el número de iteraciones"
            return sol, evaluations, None

def newton_raphson(equation, x0, tol, max_iter, error_type):
    x = sp.symbols('x')
    fx = sp.sympify(equation, locals={'ln': sp.log})
    relative_error = error_type == 'relativo'
    
    if sp.diff(fx, x).subs(x, x0) == 0:
        return "Error, derivada igual a 0", [], None

    g = (x0 - (fx.subs(x, x0) / sp.diff(fx, x).subs(x, x0))).evalf()
    e = abs(g - x0)
    
    if relative_error:
        if x0 == 0:
            return "division por cero detectada", [], None
        e = e / abs(x0)

    table = [f"it: {0} | x0: {x0} | g: {g} | e: {e}"]
    errors = [e]

    #inicio del metodo
    table = []
    if sp.diff(fx, x).subs(x,x0) == 0:
        return "Error, derivada igual a 0", table, None
    g = (x0 - (fx.subs(x,x0)/sp.diff(fx, x).subs(x,x0))).evalf()
    e = abs(g-x0)
    if relative_error:
        if x0 == 0:
            return "division por cero detectada", table, None
        e = e/abs(x0)
    row = "it: " + str(0) + "    |   x0: " + str(x0) + "  |   g: " + str(g)+ "  |   e: " + str(e)
    table.append(row)
    errors.append(e)
    ite = 1


    while e>=tol and ite<=max_iter and e!=0:
        x0=g
        if sp.diff(fx, x).subs(x,x0) == 0:
            return "Error, derivada igual a 0", table, None
        g = x0 - (fx.subs(x,x0)/sp.diff(fx, x).subs(x,x0))
        e = abs(g-x0)
        if relative_error:
            if x0 == 0:
                return "division por cero detectada", table, None
            e = e/abs(x0)
        row = "it: " + str(ite) + "    |   x0: " + str(x0) + "  |   g: " + str(g)+ "  |   e: " + str(e)
        table.append(row)
        errors.append(e)
        ite+=1

    graph_img = graf(errors)
    if e <= tol:
        sol = f"{g} es raíz de {fx} con error de {e}"
        return sol, table, graph_img
    else:
        sol = "Se excedió el número de iteraciones y no se encontró una raíz"
        return sol, table, graph_img

def punto_fijo(equation, g_equation, x0, tol, max_iter, error_type):
    x = sp.symbols('x')
    relative_error = error_type == 'relativo'
    
    try:
        fx = sp.sympify(equation, locals={'ln': sp.log})
        gx = sp.sympify(g_equation, locals={'ln': sp.log})
    except sp.SympifyError:
        return "La ecuación ingresada no es válida.", [], None

    table = []
    errors = []

    g = gx.subs(x, x0).evalf()
    e = abs(g-x0)
    if relative_error:
        if x0 == 0:
            return "division por cero detectada", table, None
        e = e/abs(x0)
    row = f"it: {0} | x0: {x0} | g: {g} | e: -"
    table.append(row)
    errors.append(e)
    ite = 1

    while e >= tol and ite <= max_iter:
        x0 = g
        g = gx.subs(x, x0).evalf()
        e = abs(g - x0)
        if relative_error:
            if x0 == 0:
                return "division por cero detectada", table, None
            e = e / abs(x0)
        row = f"it: {ite} | x0: {x0} | g: {g} | e: {e}"
        table.append(row)
        errors.append(e)
        ite += 1

    graph_img = graf(errors)
    if e <= tol:
        sol = f"{g} es raíz de {fx} con error de {e}"
        return sol, table, graph_img
    else:
        sol = "Se excedió el número de iteraciones y no se encontró una raíz"
        return sol, table, graph_img

def encontrar_ec_recta(x1, x2, y1, y2):
    if x2 - x1 != 0:
        m = (y2 - y1) / (x2 - x1)
    else:
        raise ValueError("La pendiente es indefinida (los puntos tienen la misma coordenada x).")

    b = y1 - m * x1
    rect = f"{m}*x + {b}"

    try:
        rect_equation = sp.sympify(rect)
    except sp.SympifyError:
        raise ValueError("La ecuación ingresada no es válida.")

    return rect_equation

def regla_falsa(equation, a, b, tol, error_type):
    x = sp.symbols('x')
    relative_error = error_type == 'relativo'
    
    try:
        fx = sp.sympify(equation)
    except sp.SympifyError:
        return "La ecuación ingresada no es válida.", [], None

    iteracion = 0
    table = []
    errors = []

    fa = fx.subs(x, a)
    fb = fx.subs(x, b)
    if fa == 0:
        return f"{a} es una raiz", table, None
    elif fb == 0:
        return f"{b} es una raiz", table, None
    else:
        r = encontrar_ec_recta(a, b, fa, fb)
        xm = sp.solve(r, x)[0]
        fm = fx.subs(x, xm)
        e = abs(xm - a)
        if relative_error:
            if xm == 0:
                return "division por cero detectada", table, None
            e = e / abs(xm)
        row = f"it: {iteracion}    |   xi: {a}  |  xm: {xm}  |   xf: {b}  |   e: {e}"
        table.append(row)
        errors.append(e)
        iteracion += 1

        while e >= tol and fm != 0:
            if fa * fm <= 0:
                b = xm
                fb = fm
            else:
                a = xm
                fa = fm
            r = encontrar_ec_recta(a, b, fa, fb)
            xm = sp.solve(r, x)[0]
            fm = fx.subs(x, xm)
            e = abs(xm - a)
            if relative_error:
                if xm == 0:
                    return "division por cero detectada", table, None
                e = e / abs(xm)
            row = f"it: {iteracion}    |   xi: {a}  |  xm: {xm}  |   xf: {b}  |   e: {e}"
            table.append(row)
            errors.append(e)
            iteracion += 1

        graph_img = graf(errors)
        if fm == 0:
            sol = f"{xm} es una raiz"
            return sol, table, graph_img
        else:
            sol = f"{xm} es una raiz con tolerancia de {e}"
            return sol, table, graph_img

def secante(equation, x0, x1, tol, max_iter, error_type):
    x = sp.symbols('x')
    relative_error = error_type == 'relativo'
    try:
        fx = sp.sympify(equation, locals={'ln': sp.log})
    except sp.SympifyError:
        return "La ecuación ingresada no es válida.", [], None
    
    #inicio del metodo
    table = []
    errors = []
    row = str(0) + "    |   ""  |   " + str(fx.subs(x,0))+ "  |   "
    table.append(row)
    xant = 0
    xn = 1
    g = (xn - ((fx.subs(x,xn)*(xn-xant))/(fx.subs(x,xn)-fx.subs(x,xant)))).evalf()
    if fx.subs(x,xn)-fx.subs(x,xant) == 0:
        return "division por cero detectada", table, None

    e = abs(xant-xn)
    if relative_error:
        if xn == 0:
            return "division por cero detectada", table, None
        e = e/abs(xn)
    row = str(1) + "    |   xn: " + str(xn) + "  |   g: " + str(g)+ "  |   e: " + str(e)
    table.append(row)
    errors.append(e)
    ite = 2
    #print(g)

    while e>=tol and ite<=max_iter and e !=0 and e!="NaN":
        #print("entra")
        xant = xn
        xn=g
        g = (xn - ((fx.subs(x,xn)*(xn-xant))/(fx.subs(x,xn)-fx.subs(x,xant)))).evalf()
        if fx.subs(x,xn)-fx.subs(x,xant) == 0:
            return "division por cero detectada", table, None
        e = abs(xant-xn)
        if relative_error:
            if xn == 0:
                return "division por cero detectada", table, None
            e = e/abs(xn)
        row = str(ite) + "    |  xn: " + str(xn) + "  |   g: " + str(g)+ "  |  e: " + str(e)
        table.append(row)
        errors.append(e)
        #print(e)
        ite += 1

    graph_img = graf(errors)
    if e <= tol:
        sol = f"{xn} es raíz de {fx} con error de {e}"
        return sol, table, graph_img
    else:
        sol = "Se excedió el número de iteraciones y no se encontró una raíz"
        return sol, table, graph_img
    
def raices_multiples(equation, x0, tol, max_iter, error_type):
    x = sp.symbols('x')
    relative_error = error_type == 'relativo'
    try:
        fx = sp.sympify(equation, locals={'ln': sp.log})
    except sp.SympifyError:
        return "La ecuación ingresada no es válida.", [], None

    table = []
    errors = []
    diff = sp.diff(fx, x)
    diff2 = sp.diff(diff, x)

    g = x0 - ((fx.subs(x, x0) * diff.subs(x, x0)) / (diff.subs(x, x0)**2 - fx.subs(x, x0) * diff2.subs(x, x0)))
    if diff.subs(x, x0)**2 - fx.subs(x, x0) * diff2.subs(x, x0)== 0:
        return "division por cero detectada", table, None
    e = abs(g - x0)
    if relative_error:
        if x0 == 0:
            return "division por cero detectada", table, None
        e = e / abs(x0)
    row = f"it: {0} | x0: {x0} | g: {g} | e: {e}"
    table.append(row)
    errors.append(e)
    ite = 1

    while e >= tol and ite <= max_iter and e != 0:
        x0 = g
        g = x0 - ((fx.subs(x, x0) * diff.subs(x, x0)) / (diff.subs(x, x0)**2 - fx.subs(x, x0) * diff2.subs(x, x0)))
        if diff.subs(x, x0)**2 - fx.subs(x, x0) * diff2.subs(x, x0) == 0:
            return "division por cero detectada", table, None
        e = abs(g - x0)
        if relative_error:
            if x0 == 0:
                return "division por cero detectada", table, None
            e = e / abs(x0)
        row = f"it: {ite} | x: {x0} | g: {g} | e: {e}"
        table.append(row)
        errors.append(e)
        ite += 1

    graph_img = graf(errors)
    if e <= tol:
        sol = f"{g} es raíz de {fx} con error de {e}"
        return sol, table, graph_img
    else:
        sol = "Se excedió el número de iteraciones y no se encontró una raíz"
        return sol, table, graph_img

