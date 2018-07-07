import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageDraw
import cv2
import moviepy.editor as mpy
import math
import random
# from types import SimpleNamespace

# properties
image_viewport_w = 53
image_viewport_h = 53
image_format = 'RGB'
file_name_of_twinkle_wheel = 'wheel(780x780).png'
fps = 2
rotation_per_frame = 5
file_name_of_output_mp4 = 'test'

plt.style.use('dark_background')
rain_drops = []
coord = []
n_drops = 0
delay_in_ms = 200

def main():
    # 1.    Configure these properties per source_image
    source_image_w = 780
    source_image_h = 780
    source_image_name = 'wheel(780x780).png'
    total_fibers = 2209
    norm_image_w = image_viewport_w
    norm_image_h = image_viewport_h

    # 2.    Get the viewport normalized
    viewport_image = get_viewport_image(source_image_w, source_image_h, norm_image_w, norm_image_h)
    # viewport_image.show()
    # 3.    Get the disk NOT normalized - will normalize after scale and rotate
    contour_image = get_contour_image(source_image_name, source_image_w, source_image_h)
    contour_image.show()

    contour_image_two = get_disk_image(contour_image, 5, source_image_w, source_image_h, norm_image_w, norm_image_h)
    contour_image_two.show()

    # 4.    Get the raw frames normalized
    frames, frames_rand, intensity_ratio = get_frames(viewport_image, contour_image, "0")
    # make_gif(file_name_of_output_mp4, frames, fps)
    # make_gif(file_name_of_output_mp4 + "_rand", frames_rand, fps)
    # 5.    Generate the plot 
    generate_plot(viewport_image, frames, frames_rand)




















def generate_plot(base_image, frames, frames_rand):
    total_pixels_list = get_list_of_total_of_shown_pixels(base_image)
    # print(total_pixels_list)
    n_drops = len(total_pixels_list)
    n_drops = int(math.sqrt(n_drops)) * int(math.sqrt(n_drops))
    # print("LOST PIXELS: ",len(total_pixels_list)-n_drops)
    # 6 inch square window
    v_h = 6
    v_w = 6
    intensity = 15
    pixel_distance = 0.02

    coord = []
    for x in range(int(math.sqrt(n_drops))):
        for y in range(int(math.sqrt(n_drops))):
            coord.append([pixel_distance * (x + 1), pixel_distance * (y + 1)])
    # Create new Figure and an Axes which fills it.
    fig = plt.figure(figsize=(v_w, v_h))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.axis('equal')
    ax.set_xlim(0, 1), ax.set_xticks([])
    ax.set_ylim(0, 1), ax.set_yticks([])

    # Create rain data
    rain_drops = np.zeros(n_drops, dtype=[('position', float, 2),
                                        ('size',     float, 1),
                                        ('growth',   float, 1),
                                        ('color',    float, 4)])

    # Construct the scatter which we will update during animation
    # as the raindrops develop.
    scat = ax.scatter(rain_drops['position'][:, 0], rain_drops['position'][:, 1],
                    s=rain_drops['size'], lw=0.5, edgecolors=rain_drops['color'],
                    facecolors='white')

    def update(frame_number):
        frame_index = frame_number % len(frames_rand)
        current_frame = frames_rand[frame_index]
        for drop in range(n_drops):
            coded_index = total_pixels_list[drop]
            frame_h = len(current_frame)
            frame_w = len(current_frame[0])
            random_x_point, random_y_point = get_decoded_coord(coded_index, frame_w, frame_h)
            dot_size = rain_drops['size'][drop] - int(intensity/2)
            if current_frame[random_x_point,random_y_point] != 0:
                dot_size = 1 * intensity
            rain_drops['size'][drop] = dot_size
            rain_drops['position'][drop] = coord[drop]
            rain_drops['color'][drop] = (1, 1, 1, 1)
            rain_drops['growth'][drop] = intensity/2
        # Update the scatter collection, with the new colors, sizes and positions.
        scat.set_edgecolors(rain_drops['color'])
        scat.set_sizes(rain_drops['size'])
        scat.set_offsets(rain_drops['position'])

    # Construct the animation, using the update function as the animation
    # director.
    animation = FuncAnimation(fig, update, interval=delay_in_ms)
    plt.show()





















#get contour image from input file
def get_contour_image(file_name, source_image_w, source_image_h):
    img = cv2.imread(file_name,0)
    # blur the image for better edge results
    img = cv2.medianBlur(img,5)
    # set the threshold for the contours (127,255) is good for B&W
    ret, thresh = cv2.threshold(img,127,255,0)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # define blank array similar to size of input image resolution
    blank_nparray = np.zeros((source_image_h,source_image_w,3), np.uint8)
    # draw the contours onto the nparray
    #cv2.drawContours(blank_nparray, contours, -1, (0,255,0), -1) # this only draws contour outlines
    #use for loop to fill in dark spaces
    for i in range(len(contours)-2):
        cv2.drawContours(blank_nparray, contours, len(contours)-i-1, (255,255,255), -1)
    # convert the nparray to an image using imageio
    return Image.fromarray(blank_nparray, image_format)








#get rotational images
def get_frames(norm_image, contour_image, frame_format="PIL"):
    frames = []
    frames_rand = []
    intensity_ratio = []
    # get the random mapping
    random_map_np = get_random_series(norm_image)
    viewport_np = np.array(norm_image.convert('L'))
    viewport_h = len(viewport_np)
    viewport_w = len(viewport_np[0])
    contour_np = np.array(contour_image.convert('L'))
    contour_np_h = len(contour_np)
    contour_np_w = len(contour_np[0])
    for i in range(int(360/rotation_per_frame)):
        degrees = int(i * rotation_per_frame)
        norm_disk_image = get_disk_image(contour_image, degrees, contour_np_w, contour_np_h, viewport_w, viewport_h)
        # result = Image.composite(norm_disk_image, viewport_np, viewport_np)
        binary_norm_image = np.array(norm_disk_image.convert('L'))
        new_f = np.zeros((image_viewport_w, image_viewport_h), np.uint8)
        new_f_rand = np.zeros((image_viewport_w, image_viewport_h), np.uint8)
        count_of_white_pixels = 0
        count_of_intensity = 0
        for x_point in range(image_viewport_w):
            for y_point in range(image_viewport_h):
                # print(binary_norm_image[x_point,y_point])
                if viewport_np[x_point,y_point] > 256/2:
                    count_of_white_pixels += 1
                    if binary_norm_image[x_point,y_point] > 256/2:
                        count_of_intensity += 1
                        new_f[x_point,y_point] = 255
                        coded_index = random_map_np[x_point,y_point]
                        random_x_point, random_y_point = get_decoded_coord(coded_index, image_viewport_w, image_viewport_h)
                        new_f_rand[random_x_point,random_y_point] = 255
                    else:
                        coded_index = random_map_np[x_point,y_point]
                        # print(coded_index)
                        new_f[x_point,y_point] = 0
                else:
                    new_f[x_point,y_point] = 0
                    new_f_rand[x_point,y_point] = 0
        open_cv_image = get_converted_frame(new_f, frame_format)
        open_cv_image_rand = get_converted_frame(new_f_rand, frame_format)
        frames.append(open_cv_image)
        frames_rand.append(open_cv_image_rand)
        # if i == 0:
        #     pixels = Image.fromarray(new_f_rand, 'L')
        #     pixels.show()
        #intensity_ratio = (100*count_of_intensity/count_of_white_pixels)
        intensity_ratio.append(float("{0:.2f}".format(100*count_of_intensity/count_of_white_pixels)))
        # print("INTENSITY: {0:.2f}%".format(intensity_ratio[-1]))
    # print(intensity_ratio)
    return frames, frames_rand, intensity_ratio

    # print_image():
    #     pixels = Image.fromarray(random_map_np, 'L')
    #     pixels.show()
    # print_np()
    #     print(random_map_np)












#make gif file
def make_gif(file_name, frames, fps=30):
    clips = [mpy.ImageClip(m).set_duration(1.0/fps) for m in frames]
    concat_clip = mpy.concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile("{}.mp4".format(file_name), fps=fps)















def get_list_of_total_of_shown_pixels(base_image):
    total_pixels_list = []
    np_image = np.array(base_image.convert('L'))
    count_of_white_pixels = 0
    for x_point in range(image_viewport_w):
        for y_point in range(image_viewport_h):
            # print(np_image[x_point,y_point])
            if np_image[x_point,y_point] > 256/2:
                count_of_white_pixels += 1
                index = get_encoded_coord(x_point, y_point, image_viewport_w, image_viewport_h)
                total_pixels_list.append(index)
    return total_pixels_list
















def get_image_map(base_image,random_map):
    new_f = np.zeros((image_viewport_h, image_viewport_w), int)
    np_image = np.array(base_image.convert('L'))
    counter = 0
    for x_point in range(image_viewport_w):
        for y_point in range(image_viewport_h):
            if np_image[x_point,y_point] > 256/2:
                new_f[x_point,y_point] = random_map[counter]
                counter += 1
            else:
                new_f[x_point,y_point] = 0
    return new_f














def get_random_series(base_image):
    # this will return array of all valid pixels with random positions
    total_pixels = get_list_of_total_of_shown_pixels(base_image)
    random_map = random.sample(total_pixels, len(total_pixels))
    new_map = get_image_map(base_image,random_map)
    return new_map














def get_encoded_coord(x_point, y_point, image_w, image_h):
    encoded_value = (x_point * image_h) + y_point
    return encoded_value
    











def get_decoded_coord(coded_index, image_w, image_h):
    y_point = int(coded_index % image_h)
    x_point = int((coded_index - y_point) / image_w)
    return x_point, y_point







def get_viewport_image(image_w, image_h, image_viewport_w, image_viewport_h):
    image_mask = get_mask_image(image_w, image_h)
    norm_image_mask = get_normalized_image(image_mask, image_viewport_w, image_viewport_h)
    return norm_image_mask











def get_mask_image(image_w, image_h):
    # get the first image (this is the light port)
    image1 = Image.new(image_format, (image_w, image_h))
    draw = ImageDraw.Draw(image1)
    ellipse1_tuple = (0,0,image_h,image_w)
    color1_fill = (255,255,255)
    color1_outline = (255,255,255)
    draw.ellipse(ellipse1_tuple, fill=color1_fill, outline=color1_outline)
    return image1














def get_normalized_image(base_image, image_viewport_w, image_viewport_h):
    image_frame = base_image.resize((image_viewport_w, image_viewport_h), Image.ANTIALIAS)
    return image_frame











def get_converted_frame(frame_np, frame_format):
    frame = frame_np
    if frame_format.lower() == "pillow" or frame_format.lower() == "pil" or frame_format.lower() == "rgb" or frame_format.lower() == "l":
        frame = Image.fromarray(frame, 'L')
    if frame_format.lower() == "pillow" or frame_format.lower() == "pil" or frame_format.lower() == "rgb":
        frame = np.array(open_cv_image.convert('RGB'))
    if frame_format.lower() == "pillow" or frame_format.lower() == "pil":
        frame = Image.fromarray(frame[:, :, ::-1].copy())
        # There is another way
        # img_bytes = cv2.imencode('.png', image)[1].tostring()
        # return Image.open(BytesIO(img_bytes))
    return frame










# def get_decoded_coord(coded_index, image_w, image_h):
#     y_point = int(coded_index % image_h)
#     x_point = int((coded_index - y_point) / image_w)
#     return x_point, y_point














def get_disk_image(contour_image, degrees, image_w, image_h, norm_image_w, norm_image_h):
    disk_image = contour_image.rotate(-1 * degrees)
    fiber_port_tuple = (10,image_h*(1.7/5),image_w*(2/5)+10,image_h*(3.7/5))
    disk_image = disk_image.crop(fiber_port_tuple)
    # disk_image = disk_image.resize((image_w, image_h), Image.ANTIALIAS)
    norm_disk_image = disk_image.resize((norm_image_w, norm_image_h), Image.ANTIALIAS)
    return norm_disk_image












main()