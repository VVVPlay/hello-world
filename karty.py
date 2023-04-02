import cv2
import os

def find_features(img1):
    correct_matches_dct = {} # создание словаря
    directory = 'C://Users/Student/Desktop/pythonProject32/Задача 2/images/cards/' # указание директории, откуда брать карты для сравнения
    for image in os.listdir(directory): # создание цикла для перебора карт в директории
        img2 = cv2.imread(directory+image, 0) # чтение изображения из дериктории
        orb = cv2.ORB_create() # создание ключевых точек
        kp1, des1 = orb.detectAndCompute(img1, None) # обнаружение ключевых точек данного изображения
        kp2, des2 = orb.detectAndCompute(img2, None) # обнаружение ключевых точек изображения из дериктории
        bf = cv2.BFMatcher() #
        matches = bf.knnMatch(des1, des2, k=2) # нахождение k лучших совпадений
        correct_matches = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                correct_matches.append([m])
                correct_matches_dct[image.split('.')[0]] = len(correct_matches)
    correct_matches_dct = dict(sorted(correct_matches_dct.items(), key=lambda item: item[1], reverse=True)) # добавление в словарь название нужных карт
    return list(correct_matches_dct.keys())[0]


def find_contours_of_cards(image): # функция нахождения контуров
    blurred = cv2.GaussianBlur(image, (3, 3), 0) #заблюривание изображения
    T, thresh_img = cv2.threshold(blurred, 220, 220, cv2.THRESH_BINARY) #перевод в черно-белое
    cv2.imshow('1', thresh_img) # вывод черно-белого изображения
    cv2.waitKey(0)
    (cnts, _) = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # нахождение контуров
    return cnts


def find_coordinates_of_cards(cnts, image): # функция нахождения координат карты
    cards_coordinates = {} # создание словаря, в который записываются координаты
    for i in range(0, len(cnts)): # проход по контурам
        x, y, w, h = cv2.boundingRect(cnts[i]) # нахождение рамок контуров
        if w > 50 and h > 50:
            img_crop = image[y - 15:y + h + 15, x - 15:x + w + 15] # выризаем изображение относительно контуров
            cards_name = find_features(img_crop) # возвращение названия карты, опираесь на ключевые точки
            cards_coordinates[cards_name] = (x - 15, y - 15, x + w + 15, y + h + 15) # запись в словарь название карты
    return cards_coordinates


def draw_rectangle_aroud_cards(cards_coordinates, image): # функция вывода конечного изображения
    for key, value in cards_coordinates.items():
        rec = cv2.rectangle(image, (value[0], value[1]), (value[2], value[3]), (255, 255, 0), 2) # рисование прямоудольника вокруг карты
        cv2.putText(rec, key, (value[0], value[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1) # вывод текста над квадратом
    cv2.imshow('Image', image) # вывод конечного изображения
    cv2.waitKey(0)


if __name__ == '__main__':
    main_image = cv2.imread('base.JPG') # чтение данного изображения
    gray_main_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY) # перевод данного изображения в оттенки серого
    contours = find_contours_of_cards(gray_main_image) # вызов функции нахождения контуров
    cards_location = find_coordinates_of_cards(contours, gray_main_image) # вызов функции нахождения координат карты
    draw_rectangle_aroud_cards(cards_location, main_image) # вызов функции вывода конечного изображения
