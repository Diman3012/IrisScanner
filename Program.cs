using OpenCvSharp;
using System.Net.Http;

const string FaceCascadeUrl =
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml";

const string EyeCascadeUrl =
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_eye.xml";

// ----------------------------------------------------
// Пути
// ----------------------------------------------------

string modelDir =
    @"C:\IrisScannerModels";

Directory.CreateDirectory(modelDir);

string faceCascadePath =
    Path.Combine(
        modelDir,
        "haarcascade_frontalface_default.xml");

string eyeCascadePath =
    Path.Combine(
        modelDir,
        "haarcascade_eye.xml");

// ----------------------------------------------------
// Загрузка моделей
// ----------------------------------------------------

await DownloadIfMissingAsync(
    FaceCascadeUrl,
    faceCascadePath);

await DownloadIfMissingAsync(
    EyeCascadeUrl,
    eyeCascadePath);

Console.WriteLine("Модели готовы.");

// ----------------------------------------------------
// Детекторы
// ----------------------------------------------------

using var faceDetector =
    new CascadeClassifier(faceCascadePath);

using var eyeDetector =
    new CascadeClassifier(eyeCascadePath);

// ----------------------------------------------------
// Камера
// ----------------------------------------------------

using var camera =
    new VideoCapture(
        0,
        VideoCaptureAPIs.DSHOW);

if (!camera.IsOpened())
{
    Console.WriteLine(
        "Не удалось открыть камеру.");

    return;
}

camera.Set(
    VideoCaptureProperties.FrameWidth,
    1280);

camera.Set(
    VideoCaptureProperties.FrameHeight,
    720);

// ----------------------------------------------------
// Mat
// ----------------------------------------------------

using var frame =
    new Mat();

using var gray =
    new Mat();

using var display =
    new Mat();

using var blurred =
    new Mat();

// ----------------------------------------------------
// Окно
// ----------------------------------------------------

using var window =
    new Window(
        "IRIS SCANNER - ESC",
        WindowFlags.AutoSize);

// ----------------------------------------------------
// Состояние сканирования
// ----------------------------------------------------

double scanProgress = 0;

bool irisFound = false;

// ====================================================
// НАСТРОЙКИ ЗУМА
// ====================================================

// Сила приближения радужки.
// 4 = слабее
// 6 = средне
// 8 = сильнее
int zoom = 6;

// Размер картинки увеличенной радужки.
const int zoomDisplaySize = 500;

// ====================================================
// ГЛАВНЫЙ ЦИКЛ
// ====================================================

while (true)
{
    if (!camera.Read(frame) ||
        frame.Empty())
    {
        break;
    }

    // =================================================
    // ЦВЕТНОЙ DISPLAY
    // =================================================

    // ВАЖНО:
    // Здесь больше НЕ переводим display в Gray.
    // Поэтому основное изображение камеры остаётся цветным.

    frame.CopyTo(display);

    irisFound = false;

    // =================================================
    // ЧЁРНО-БЕЛОЕ ИЗОБРАЖЕНИЕ ДЛЯ ДЕТЕКТОРОВ
    // =================================================

    Cv2.CvtColor(
        frame,
        gray,
        ColorConversionCodes.BGR2GRAY);

    Cv2.EqualizeHist(
        gray,
        gray);

    // =================================================
    // ПОИСК ЛИЦА
    // =================================================

    Rect[] faces =
        faceDetector.DetectMultiScale(
            gray,
            1.1,
            5,
            HaarDetectionTypes.ScaleImage,
            new Size(150, 150));

    if (faces.Length > 0)
    {
        // Берём самое большое лицо.
        Rect face =
            faces
                .OrderByDescending(
                    f => f.Width * f.Height)
                .First();

        // =================================================
        // РАМКА ЛИЦА
        // =================================================

        Cv2.Rectangle(
            display,
            face,
            Scalar.LimeGreen,
            2,
            LineTypes.AntiAlias);

        // =================================================
        // ОБЛАСТЬ ГЛАЗ
        // =================================================

        int eyeRegionHeight =
            Math.Max(
                1,
                (int)(face.Height * 0.60));

        var eyeRegion =
            new Rect(
                face.X,
                face.Y,
                face.Width,
                eyeRegionHeight);

        using var faceTop =
            new Mat(
                gray,
                eyeRegion);

        // =================================================
        // ПОИСК ГЛАЗ
        // =================================================

        Rect[] eyes =
            eyeDetector.DetectMultiScale(
                faceTop,
                1.08,
                6,
                HaarDetectionTypes.ScaleImage,
                new Size(35, 35));

        // =================================================
        // ОБРАБОТКА ГЛАЗ
        // =================================================

        foreach (Rect eye in eyes)
        {
            int ex =
                eyeRegion.X + eye.X;

            int ey =
                eyeRegion.Y + eye.Y;

            // ---------------------------------------------
            // ВНУТРЕННЯЯ ЧАСТЬ ГЛАЗА
            // ---------------------------------------------

            int innerX =
                ex + eye.Width / 10;

            int innerY =
                ey + eye.Height / 6;

            int innerW =
                Math.Max(
                    10,
                    eye.Width * 8 / 10);

            int innerH =
                Math.Max(
                    10,
                    eye.Height * 2 / 3);

            if (innerX < 0 ||
                innerY < 0 ||
                innerX + innerW > gray.Width ||
                innerY + innerH > gray.Height)
            {
                continue;
            }

            var inner =
                new Rect(
                    innerX,
                    innerY,
                    innerW,
                    innerH);

            // ---------------------------------------------
            // РАМКА ГЛАЗА
            // ---------------------------------------------

            Cv2.Rectangle(
                display,
                new Rect(
                    ex,
                    ey,
                    eye.Width,
                    eye.Height),
                Scalar.Yellow,
                2,
                LineTypes.AntiAlias);

            // ---------------------------------------------
            // ВЫРЕЗАЕМ ГЛАЗ
            // ---------------------------------------------

            using var roi =
                new Mat(
                    gray,
                    inner);

            Cv2.GaussianBlur(
                roi,
                blurred,
                new Size(7, 7),
                1.5);

            // =================================================
            // ПОИСК РАДУЖКИ
            // =================================================

            CircleSegment[] circles =
                Cv2.HoughCircles(
                    blurred,
                    HoughModes.Gradient,
                    dp: 1.2,
                    minDist: Math.Max(
                        15,
                        inner.Width / 2),
                    param1: 100,
                    param2: 14,
                    minRadius: Math.Max(
                        4,
                        inner.Height / 6),
                    maxRadius: Math.Max(
                        6,
                        inner.Height / 2));

            if (circles.Length == 0)
            {
                continue;
            }

            // =================================================
            // ВЫБИРАЕМ ЛУЧШУЮ РАДУЖКУ
            // =================================================

            CircleSegment iris =
                circles
                    .OrderByDescending(
                        c => c.Radius)
                    .First();

            int centerX =
                inner.X +
                (int)iris.Center.X;

            int centerY =
                inner.Y +
                (int)iris.Center.Y;

            int radius =
                Math.Max(
                    3,
                    (int)iris.Radius);

            var center =
                new Point(
                    centerX,
                    centerY);

            irisFound = true;

            // =================================================
            // КРУГ РАДУЖКИ
            // =================================================

            Cv2.Circle(
                display,
                center,
                radius,
                Scalar.Red,
                3,
                LineTypes.AntiAlias);

            // =================================================
            // ВНУТРЕННИЙ КРУГ
            // =================================================

            Cv2.Circle(
                display,
                center,
                Math.Max(
                    2,
                    radius / 2),
                Scalar.Cyan,
                2,
                LineTypes.AntiAlias);

            // =================================================
            // ЦЕНТР РАДУЖКИ
            // =================================================

            Cv2.Circle(
                display,
                center,
                4,
                Scalar.White,
                -1,
                LineTypes.AntiAlias);

            // =================================================
            // РАДИАЛЬНАЯ СЕТКА
            // =================================================

            for (int i = 0; i < 12; i++)
            {
                double gridAngle =
                    i *
                    Math.PI /
                    6.0;

                int x1 =
                    center.X +
                    (int)(
                        Math.Cos(gridAngle)
                        * radius);

                int y1 =
                    center.Y +
                    (int)(
                        Math.Sin(gridAngle)
                        * radius);

                int x2 =
                    center.X -
                    (int)(
                        Math.Cos(gridAngle)
                        * radius);

                int y2 =
                    center.Y -
                    (int)(
                        Math.Sin(gridAngle)
                        * radius);

                Cv2.Line(
                    display,
                    new Point(
                        x1,
                        y1),
                    new Point(
                        x2,
                        y2),
                    Scalar.Cyan,
                    1,
                    LineTypes.AntiAlias);
            }

            // =================================================
            // ВРАЩАЮЩИЙСЯ ЛУЧ
            // =================================================

            long time =
                Environment.TickCount64;

            double scanAngle =
                (time % 2000) /
                2000.0 *
                Math.PI *
                2.0;

            int scanX =
                center.X +
                (int)(
                    Math.Cos(scanAngle)
                    * radius);

            int scanY =
                center.Y +
                (int)(
                    Math.Sin(scanAngle)
                    * radius);

            Cv2.Line(
                display,
                center,
                new Point(
                    scanX,
                    scanY),
                Scalar.Red,
                3,
                LineTypes.AntiAlias);

            // =================================================
            // НАДПИСЬ IRIS
            // =================================================

            Cv2.PutText(
                display,
                "IRIS",
                new Point(
                    center.X + radius + 10,
                    center.Y),
                HersheyFonts.HersheySimplex,
                0.6,
                Scalar.Red,
                2,
                LineTypes.AntiAlias);

            // =================================================
            // АВТОМАТИЧЕСКИЙ ЦВЕТНОЙ ЗУМ РАДУЖКИ
            // =================================================

            // Размер области, которую вырезаем вокруг радужки.
            int cropSize =
                radius * zoom;

            // Минимальный размер области.
            cropSize =
                Math.Max(
                    cropSize,
                    120);

            // ---------------------------------------------
            // ЛЕВЫЙ ВЕРХНИЙ УГОЛ ОБЛАСТИ
            // ---------------------------------------------

            int cropX =
                center.X -
                cropSize / 2;

            int cropY =
                center.Y -
                cropSize / 2;

            // ---------------------------------------------
            // НЕ ВЫХОДИМ ЗА ГРАНИЦЫ КАДРА
            // ---------------------------------------------

            cropX =
                Math.Clamp(
                    cropX,
                    0,
                    frame.Width - 1);

            cropY =
                Math.Clamp(
                    cropY,
                    0,
                    frame.Height - 1);

            // ---------------------------------------------
            // КОРРЕКТИРУЕМ РАЗМЕР
            // ---------------------------------------------

            int cropWidth =
                Math.Min(
                    cropSize,
                    frame.Width - cropX);

            int cropHeight =
                Math.Min(
                    cropSize,
                    frame.Height - cropY);

            // ---------------------------------------------
            // ПРОВЕРКА
            // ---------------------------------------------

            if (cropWidth > 20 &&
                cropHeight > 20)
            {
                var irisCropRect =
                    new Rect(
                        cropX,
                        cropY,
                        cropWidth,
                        cropHeight);

                // ВАЖНО:
                // Берём область именно из frame,
                // поэтому зум остаётся ЦВЕТНЫМ.

                using var irisCrop =
                    new Mat(
                        frame,
                        irisCropRect);

                using var irisZoom =
                    new Mat();

                // -----------------------------------------
                // УВЕЛИЧЕНИЕ
                // -----------------------------------------

                Cv2.Resize(
                    irisCrop,
                    irisZoom,
                    new Size(
                        zoomDisplaySize,
                        zoomDisplaySize),
                    0,
                    0,
                    InterpolationFlags.Cubic);

                // -----------------------------------------
                // РАМКА
                // -----------------------------------------

                Cv2.Rectangle(
                    irisZoom,
                    new Rect(
                        3,
                        3,
                        zoomDisplaySize - 6,
                        zoomDisplaySize - 6),
                    Scalar.Cyan,
                    3);

                // -----------------------------------------
                // ЦЕНТР РАДУЖКИ НА ЗУМЕ
                // -----------------------------------------

                int zoomCenterX =
                    (center.X - cropX) *
                    zoomDisplaySize /
                    cropWidth;

                int zoomCenterY =
                    (center.Y - cropY) *
                    zoomDisplaySize /
                    cropHeight;

                var zoomCenter =
                    new Point(
                        zoomCenterX,
                        zoomCenterY);

                // -----------------------------------------
                // РАДИУС НА ЗУМЕ
                // -----------------------------------------

                int zoomRadius =
                    Math.Max(
                        5,
                        radius *
                        zoomDisplaySize /
                        cropWidth);

                // -----------------------------------------
                // КРУГ РАДУЖКИ НА ЗУМЕ
                // -----------------------------------------

                Cv2.Circle(
                    irisZoom,
                    zoomCenter,
                    zoomRadius,
                    Scalar.Red,
                    3,
                    LineTypes.AntiAlias);

                // -----------------------------------------
                // ЦЕНТР
                // -----------------------------------------

                Cv2.Circle(
                    irisZoom,
                    zoomCenter,
                    5,
                    Scalar.White,
                    -1,
                    LineTypes.AntiAlias);

                // -----------------------------------------
                // СЕТКА НА ЗУМЕ
                // -----------------------------------------

                for (int i = 0; i < 12; i++)
                {
                    double zoomGridAngle =
                        i *
                        Math.PI /
                        6.0;

                    int zx1 =
                        zoomCenter.X +
                        (int)(
                            Math.Cos(
                                zoomGridAngle)
                            * zoomRadius);

                    int zy1 =
                        zoomCenter.Y +
                        (int)(
                            Math.Sin(
                                zoomGridAngle)
                            * zoomRadius);

                    int zx2 =
                        zoomCenter.X -
                        (int)(
                            Math.Cos(
                                zoomGridAngle)
                            * zoomRadius);

                    int zy2 =
                        zoomCenter.Y -
                        (int)(
                            Math.Sin(
                                zoomGridAngle)
                            * zoomRadius);

                    Cv2.Line(
                        irisZoom,
                        new Point(
                            zx1,
                            zy1),
                        new Point(
                            zx2,
                            zy2),
                        Scalar.Cyan,
                        1,
                        LineTypes.AntiAlias);
                }

                // -----------------------------------------
                // ЗАГОЛОВОК
                // -----------------------------------------

                Cv2.PutText(
                    irisZoom,
                    "IRIS ZOOM",
                    new Point(
                        15,
                        32),
                    HersheyFonts.HersheySimplex,
                    0.8,
                    Scalar.Cyan,
                    2,
                    LineTypes.AntiAlias);

                // -----------------------------------------
                // РАЗМЕР ЗУМА
                // -----------------------------------------

                Cv2.PutText(
                    irisZoom,
                    $"ZOOM x{zoom}",
                    new Point(
                        15,
                        62),
                    HersheyFonts.HersheySimplex,
                    0.55,
                    Scalar.White,
                    2,
                    LineTypes.AntiAlias);

                // =================================================
                // ПОКАЗЫВАЕМ ЗУМ СПРАВА СВЕРХУ
                // =================================================

                int zoomX =
                    display.Width -
                    zoomDisplaySize -
                    20;

                int zoomY =
                    20;

                // ---------------------------------------------
                // ЧЁРНАЯ ПОДЛОЖКА ПОД ЗУМ
                // ---------------------------------------------

                if (zoomX >= 0 &&
                    zoomY >= 0 &&
                    zoomX + zoomDisplaySize <= display.Width &&
                    zoomY + zoomDisplaySize <= display.Height)
                {
                    var zoomArea =
                        new Rect(
                            zoomX,
                            zoomY,
                            zoomDisplaySize,
                            zoomDisplaySize);

                    using var target =
                        new Mat(
                            display,
                            zoomArea);

                    irisZoom.CopyTo(
                        target);
                }
            }

            // Обрабатываем только один глаз.
            break;
        }
    }

    // =================================================
    // ПРОГРЕСС СКАНИРОВАНИЯ
    // =================================================

    if (irisFound)
    {
        scanProgress += 1.5;

        if (scanProgress > 100)
            scanProgress = 100;
    }
    else
    {
        scanProgress -= 1.0;

        if (scanProgress < 0)
            scanProgress = 0;
    }

    // =================================================
    // UI
    // =================================================

    Cv2.PutText(
        display,
        "IRIS SCANNER",
        new Point(
            20,
            35),
        HersheyFonts.HersheySimplex,
        0.9,
        Scalar.White,
        2,
        LineTypes.AntiAlias);

    // =================================================
    // СТАТУС
    // =================================================

    if (!irisFound)
    {
        Cv2.PutText(
            display,
            "LOOK AT CAMERA",
            new Point(
                20,
                70),
            HersheyFonts.HersheySimplex,
            0.7,
            Scalar.White,
            2,
            LineTypes.AntiAlias);
    }
    else
    {
        Cv2.PutText(
            display,
            "IRIS DETECTED",
            new Point(
                20,
                70),
            HersheyFonts.HersheySimplex,
            0.7,
            Scalar.LimeGreen,
            2,
            LineTypes.AntiAlias);
    }

    // =================================================
    // ПОЛОСА ПРОГРЕССА
    // =================================================

    int barX = 20;
    int barY = 105;
    int barWidth = 350;
    int barHeight = 22;

    Cv2.Rectangle(
        display,
        new Rect(
            barX,
            barY,
            barWidth,
            barHeight),
        Scalar.White,
        2);

    int progressWidth =
        (int)(
            barWidth *
            scanProgress /
            100.0);

    if (progressWidth > 0)
    {
        Cv2.Rectangle(
            display,
            new Rect(
                barX,
                barY,
                progressWidth,
                barHeight),
            Scalar.Cyan,
            -1);
    }

    // =================================================
    // ТЕКСТ
    // =================================================

    Cv2.PutText(
        display,
        $"SCAN {scanProgress:0}%",
        new Point(
            20,
            160),
        HersheyFonts.HersheySimplex,
        0.7,
        Scalar.White,
        2,
        LineTypes.AntiAlias);

    Cv2.PutText(
        display,
        "Move closer to camera",
        new Point(
            20,
            195),
        HersheyFonts.HersheySimplex,
        0.6,
        Scalar.White,
        2,
        LineTypes.AntiAlias);

    Cv2.PutText(
        display,
        "IRIS ZOOM: x6",
        new Point(
            20,
            225),
        HersheyFonts.HersheySimplex,
        0.6,
        Scalar.Cyan,
        2,
        LineTypes.AntiAlias);

    Cv2.PutText(
        display,
        "ESC - EXIT",
        new Point(
            20,
            255),
        HersheyFonts.HersheySimplex,
        0.6,
        Scalar.White,
        2,
        LineTypes.AntiAlias);

    // =================================================
    // ПОКАЗ
    // =================================================

    window.ShowImage(
        display);

    int key =
        Cv2.WaitKey(1);

    if (key == 27)
        break;
}

// =====================================================
// ЗАГРУЗКА ФАЙЛА
// =====================================================

static async Task DownloadIfMissingAsync(
    string url,
    string path)
{
    if (File.Exists(path))
        return;

    Console.WriteLine(
        $"Загрузка: {Path.GetFileName(path)}");

    using var client =
        new HttpClient();

    client.Timeout =
        TimeSpan.FromMinutes(5);

    byte[] data =
        await client.GetByteArrayAsync(url);

    await File.WriteAllBytesAsync(
        path,
        data);

    Console.WriteLine(
        $"Готово: {Path.GetFileName(path)}");
}