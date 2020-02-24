# -*- coding: utf-8 -*-
from constants import *
import scipy.integrate

from manimlib.imports import *

USE_ALMOST_FOURIER_BY_DEFAULT = True
NUM_SAMPLES_FOR_FFT = 1000
DEFAULT_COMPLEX_TO_REAL_FUNC = lambda z : z.real


def get_fourier_graph(
    axes, time_func, t_min, t_max,
    n_samples = NUM_SAMPLES_FOR_FFT,
    complex_to_real_func = lambda z : z.real,
    color = RED,
    ):
    # N = n_samples
    # T = time_range/n_samples
    time_range = float(t_max - t_min)
    time_step_size = time_range/n_samples
    time_samples = np.vectorize(time_func)(np.linspace(t_min, t_max, n_samples))
    fft_output = np.fft.fft(time_samples)
    frequencies = np.linspace(0.0, n_samples/(2.0*time_range), n_samples//2)
    #  #Cycles per second of fouier_samples[1]
    # (1/time_range)*n_samples
    # freq_step_size = 1./time_range
    graph = VMobject()
    graph.set_points_smoothly([
        axes.coords_to_point(
            x, complex_to_real_func(y)/n_samples,
        )
        for x, y in zip(frequencies, fft_output[:n_samples//2])
    ])
    graph.set_color(color)
    f_min, f_max = [
        axes.x_axis.point_to_number(graph.points[i])
        for i in (0, -1)
    ]
    graph.underlying_function = lambda f : axes.y_axis.point_to_number(
        graph.point_from_proportion((f - f_min)/(f_max - f_min))
    )
    return graph

def get_fourier_transform(
    func, t_min, t_max, 
    complex_to_real_func = DEFAULT_COMPLEX_TO_REAL_FUNC,
    use_almost_fourier = USE_ALMOST_FOURIER_BY_DEFAULT,
    **kwargs ##Just eats these
    ):
    scalar = 1./(t_max - t_min) if use_almost_fourier else 1.0
    def fourier_transform(f):
        z = scalar*scipy.integrate.quad(
            lambda t : func(t)*np.exp(complex(0, -TAU*f*t)),
            t_min, t_max
        )[0]
        return complex_to_real_func(z)
    return fourier_transform










class Block(Rectangle):
    CONFIG = {
        "mass": 1,
        "velocity": 0,
        "width": 0.1,
        "height": 8.0,
        "fill_opacity": 1,
        "stroke_width": 1,
        "stroke_color": WHITE,
        "fill_color": None,
        
    
    }

    def __init__(self, **kwargs):
        digest_config(self, kwargs)
        Rectangle.__init__(self, side_length=self.width, **kwargs)
       

class AirMolecule(Circle):
    CONFIG = {
        "velocity": 0,
        "radius": 0.1,
        "fill_opacity": 1,
        "stroke_width": 0.1,
        "stroke_color": BLUE,
        "fill_color": BLUE,
        
    
    }

    def __init__(self, **kwargs):
        digest_config(self, kwargs)
        Circle.__init__(self, **kwargs)

class SlidingBlocks(VGroup):
    CONFIG = {
        "block_config": {
            "distance": 1,
            "mass": 1,
            "velocity": 0,
        },
        "membrane":-4.3,
        "floor_pos":-2,
        "amplitude": 0.2,
        "frequency":5,
        "timer":0,
    }

    def __init__(self, scene, **kwargs):
        VGroup.__init__(self, **kwargs)
        self.scene = scene
        self.block = self.get_block(**self.block_config)
        
        self.point_field = self.get_points()
        
        self.add(
            self.block,
        )
        self.add_updater(self.__class__.update_positions)



    def get_block(self, distance, **kwargs):
        block = Block(**kwargs)
        block.move_to(
            self.floor_pos * UP +
            (self.membrane + distance) * LEFT,
            DL,
        )
        return block

    def get_points(self,**kwargs):
        points = [x*RIGHT+y*UP
            for x in np.arange(-2,6,0.25)
            for y in np.arange(-4,4,0.25)
            ]
        point_field=[]
        for point in points:
            circle=AirMolecule().shift(point)
            circle.x = point[0]
            circle.y = point[1]
            point_field.append(circle)
            self.add(circle) 
        return point_field
            

  
    def update_positions(self, dt):
        block= self.block
        floor_y = self.floor_pos
        #point_field = self.point_field
        self.timer +=dt
        ps_block= self.amplitude* np.sin(self.frequency* self.timer)
        block.move_to(
                (self.membrane+ ps_block) * RIGHT +
                floor_y * UP,      
                DL,
            )
            
        for point in self.point_field:
            ps_point = point.x + self.amplitude*np.sin(self.frequency * (self.timer + point.x))
            point.move_to(
                (point.x+ ps_point) * RIGHT +
                point.y * UP,      
                DL,
            )



class AirPressure(Scene):
    CONFIG = {
        "sliding_blocks_config": {
            "block_config": {
                "mass": 1e0,
                "velocity": -2,
            }
        },
        "wait_time": 15,
        "frequency" : 1,
        "amplitude":2,
        "A_color" : YELLOW,
        "D_color" : PINK,
        "F_color" : TEAL,
        "C_color" : RED,
        "sum_color" : GREEN,
        "equilibrium_height" : 1.5,
        "plane_kwargs" : {},
    }
    def setup(self):
        plane = NumberPlane(**self.plane_kwargs)
        self.track_time()
        self.add_blocks_and_points()
        #self.add_points()

        #self.add_speaker()
        

    def add_blocks_and_points(self):
        self.blocks = SlidingBlocks(self, **self.sliding_blocks_config)
        self.add(self.blocks)

    def track_time(self):
        time_tracker = ValueTracker()
        time_tracker.add_updater(lambda m, dt: m.increment_value(dt))
        self.add(time_tracker)
        self.get_time = time_tracker.get_value

    def add_speaker(self):
        self.speaker = SVGMobject(file_name = "speaker.svg")
        self.add(self.speaker)

    def add_points(self):
        self.points = [x*RIGHT+y*UP
            for x in np.arange(-4,6,0.5)
            for y in np.arange(-5,5,0.5)
            ]
               

        self.point_field = [] 
        for point in self.points:
            circle=AirMolecule().shift(point)
            circle.x = point[0]
            circle.y = point[1]
            self.point_field.append(circle)
            self.add(circle) 
            

    def construct(self):
        #self.show_sine()
        self.wait(5)
        

    def shift_circles(self):
        for point in self.point_field:
            point.shift(RIGHT)


    def show_sine(self):
        axes = Axes(
            y_min = -2, y_max = 2,
            x_min = 0, x_max = 14,
            number_line_config = {"include_tip" : False},
        )
        axes.stretch_to_fit_height(2)
        axes.shift(4*LEFT)
        axes.shift(DOWN)

       
        graph = self.get_wave_graph(axes)
        func = graph.underlying_function
        graph.set_color(self.A_color)

        
        self.play(
            ShowCreation(graph, run_time = 6, rate_func=linear),
            
        )

        
        self.wait(self.wait_time)

    def get_wave_graph(self, axes):
        tail_len = 1.0
        x_min, x_max = axes.x_min, axes.x_max
        def func(x):
            value = self.amplitude*np.cos(2*np.pi*self.frequency*x)
            if x - x_min < tail_len:
                value *= smooth((x-x_min)/tail_len)
            if x_max - x < tail_len:
                value *= smooth((x_max - x )/tail_len)
            return value + self.equilibrium_height
        ngp = 2*(x_max - x_min)*self.frequency + 1
        graph = axes.get_graph(func, num_graph_points = int(ngp))
        return graph
        
        









 

class WhatIsATone(Scene):
    CONFIG = {
        "A_frequency" : 2.1,
        "A_color" : YELLOW,
        "D_color" : PINK,
        "F_color" : TEAL,
        "C_color" : RED,
        "sum_color" : GREEN,
        "equilibrium_height" : 1.5,
    }

    def construct(self):
        self.show_sine()

    def show_sine(self):
        axes = Axes(
            y_min = -2, y_max = 2,
            x_min = 0, x_max = 14,
            number_line_config = {"include_tip" : False},
        )
        axes.stretch_to_fit_height(2)
        axes.to_corner(2*LEFT)
        #axes.shift(LARGE_BUFF*DOWN)
       
        frequency = self.A_frequency
        graph = self.get_wave_graph(frequency, axes)
        func = graph.underlying_function
        graph.set_color(self.A_color)

        
        self.play(
            ShowCreation(graph, run_time = 6, rate_func=linear),
            #ShowCreation(equilibrium_line),
        )
        #axes.add(equilibrium_line)
        
        self.wait()

    def speaker(self):
        x = 2.85
        frequency=3
        axes = Axes(
            y_min = -2, y_max = 2,
            x_min = 0, x_max = 10,
            number_line_config = {"include_tip" : False},
        )
        rect = Rectangle(
            height = 2.5*FRAME_Y_RADIUS,
            width = MED_SMALL_BUFF,
            stroke_width = 0,
            fill_color = YELLOW,
            fill_opacity = 0.4
        )
        x_min, x_max = axes.x_min, axes.x_max
        func = np.sin
        ngp = 2*(x_max - x_min)*frequency + 1
        graph = Axes.get_graph(func,self.equilibrium_height, num_graph_points = int(ngp))

        

        self.play(MoveAlongPath(rect,graph))
        self.wait()










    def get_wave_graph(self, frequency, axes):
        tail_len = 3.0
        x_min, x_max = axes.x_min, axes.x_max
        def func(x):
            value = 0.7*np.cos(2*np.pi*frequency*x)
            if x - x_min < tail_len:
                value *= smooth((x-x_min)/tail_len)
            if x_max - x < tail_len:
                value *= smooth((x_max - x )/tail_len)
            return value + self.equilibrium_height
        ngp = 2*(x_max - x_min)*frequency + 1
        graph = axes.get_graph(func, num_graph_points = int(ngp))
        return graph
        
        




























class AddingPureFrequencies(Scene):
    CONFIG = {
        "A_frequency" : 2.1,
        "A_color" : YELLOW,
        "D_color" : PINK,
        "F_color" : TEAL,
        "C_color" : RED,
        "sum_color" : GREEN,
        "equilibrium_height" : 1.5,
    }
    def construct(self):
        #self.add_speaker()
        self.play_A440()
        self.measure_air_pressure()
        self.play_lower_pitch()
        self.play_mix()
        self.separate_out_parts()
        self.draw_sum_at_single_point()
        self.draw_full_sum()
        #self.add_more_notes()

    def add_speaker(self):
        speaker = SVGMobject(file_name = "speaker")
        speaker.to_edge(DOWN)

        self.add(speaker)
        self.speaker = speaker

    def play_A440(self):
        A_label = TextMobject("A440")
        A_label.set_color(self.A_color)
        A_label.to_edge(DOWN)
        A_label.move_to(UP)

        self.play(
            FadeIn(A_label),
            
        )

        self.set_variables_as_attrs(A_label)

    def measure_air_pressure(self):
        #randy = self.pi_creature
        axes = Axes(
            y_min = -2, y_max = 2,
            x_min = 0, x_max = 10,
            number_line_config = {"include_tip" : False},
        )
        axes.stretch_to_fit_height(2)
        axes.to_corner(UP+LEFT)
        axes.shift(LARGE_BUFF*DOWN)
        eh = self.equilibrium_height
        equilibrium_line = DashedLine(
            axes.coords_to_point(0, eh),
            axes.coords_to_point(axes.x_max, eh),
            stroke_width = 2,
            stroke_color = LIGHT_GREY
        )

        frequency = self.A_frequency
        graph = self.get_wave_graph(frequency, axes)
        func = graph.underlying_function
        graph.set_color(self.A_color)
        pressure = TextMobject("Pressure")
        time = TextMobject("Time")
        for label in pressure, time:
            label.scale_in_place(0.8)
        pressure.next_to(axes.y_axis, UP)
        pressure.to_edge(LEFT, buff = MED_SMALL_BUFF)
        time.next_to(axes.x_axis.get_right(), DOWN+LEFT)
        axes.labels = VGroup(pressure, time)

        n = 10
        brace = Brace(Line(
            axes.coords_to_point(n/frequency, func(n/frequency)),
            axes.coords_to_point((n+1)/frequency, func((n+1)/frequency)),
        ), UP)
        words = brace.get_text("Imagine 440 per second", buff = SMALL_BUFF)
        words.scale(0.8, about_point = words.get_bottom())

        self.play(
            FadeIn(pressure),
            ShowCreation(axes.y_axis)
        )
        self.play(
            Write(time),
            ShowCreation(axes.x_axis)
        )
        self.play(
            ShowCreation(graph, run_time = 4, rate_func=linear),
            ShowCreation(equilibrium_line),
        )
        axes.add(equilibrium_line)
        self.play(
            #randy.change, "erm", graph,
            GrowFromCenter(brace),
            Write(words)
        )
        self.wait()
        graph.save_state()
        self.play(
            FadeOut(brace),
            FadeOut(words),
            VGroup(axes, graph, axes.labels).shift, 0.8*UP,
            graph.fade, 0.85,
            graph.shift, 0.8*UP,
        )

        graph.saved_state.move_to(graph)
        self.set_variables_as_attrs(axes, A_graph = graph)

    def play_lower_pitch(self):
        axes = self.axes
        #randy = self.pi_creature

        frequency = self.A_frequency*(2.0/3.0)
        graph = self.get_wave_graph(frequency, axes)
        graph.set_color(self.D_color)

        D_label = TextMobject("D294")
        D_label.set_color(self.D_color)
        #D_label.move_to(self.A_label)

        self.play(
            #FadeOut(self.A_label),
            GrowFromCenter(D_label),
        )
        self.play(
            ShowCreation(graph, run_time = 4, rate_func=linear),
            #randy.change, "happy",
            n_circles = 6,
        )
        self.wait(2)

        self.set_variables_as_attrs(
            D_label,
            D_graph = graph
        )

    def play_mix(self):
        self.A_graph.restore()
        self.play(
            #self.get_broadcast_animation(n_circles = 6),
            *[
                ShowCreation(graph, run_time = 4, rate_func=linear)
                for graph in (self.A_graph, self.D_graph)
            ]
        )
        self.wait()

    def separate_out_parts(self):
        axes = self.axes
        #speaker = self.speaker
        #randy = self.pi_creature

        A_axes = axes.deepcopy()
        A_graph = self.A_graph
        A_label = self.A_label
        D_axes = axes.deepcopy()
        D_graph = self.D_graph
        D_label = self.D_label
        movers = [A_axes, A_graph, A_label, D_axes, D_graph, D_label]
        for mover in movers:
            mover.generate_target()
        D_target_group = VGroup(D_axes.target, D_graph.target)
        A_target_group = VGroup(A_axes.target, A_graph.target)
        D_target_group.next_to(axes, DOWN, MED_LARGE_BUFF)
        A_target_group.next_to(D_target_group, DOWN, MED_LARGE_BUFF)
        A_label.fade(1)
        A_label.target.next_to(A_graph.target, UP)
        D_label.target.next_to(D_graph.target, UP)

        self.play(*it.chain(
            list(map(MoveToTarget, movers)),
            
        ))
        self.wait()

        self.set_variables_as_attrs(A_axes, D_axes)

    def draw_sum_at_single_point(self):
        axes = self.axes
        A_axes = self.A_axes
        D_axes = self.D_axes
        A_graph = self.A_graph
        D_graph = self.D_graph

        x = 2.85
        A_line = self.get_A_graph_v_line(x)
        D_line = self.get_D_graph_v_line(x)
        lines = VGroup(A_line, D_line)
        sum_lines = lines.copy()
        sum_lines.generate_target()
        self.stack_v_lines(x, sum_lines.target)

        top_axes_point = axes.coords_to_point(x, self.equilibrium_height)
        x_point = np.array(top_axes_point)
        x_point[1] = 0
        v_line = Line(UP, DOWN).scale(FRAME_Y_RADIUS).move_to(x_point)

        self.revert_to_original_skipping_status()
        self.play(GrowFromCenter(v_line))
        self.play(FadeOut(v_line))
        self.play(*list(map(ShowCreation, lines)))
        self.wait()
        self.play(MoveToTarget(sum_lines, path_arc = np.pi/4))
        self.wait(2)
        # self.play(*[
        #     Transform(
        #         line, 
        #         VectorizedPoint(axes.coords_to_point(0, self.equilibrium_height)),
        #         remover = True
        #     )
        #     for line, axes in [
        #         (A_line, A_axes),
        #         (D_line, D_axes),
        #         (sum_lines, axes),
        #     ]
        # ])
        self.lines_to_fade = VGroup(A_line, D_line, sum_lines)

    def draw_full_sum(self):
        axes = self.axes

        def new_func(x):
            result = self.A_graph.underlying_function(x)
            result += self.D_graph.underlying_function(x)
            result -= self.equilibrium_height
            return result

        sum_graph = axes.get_graph(new_func)
        sum_graph.set_color(self.sum_color)
        thin_sum_graph = sum_graph.copy().fade()

        A_graph = self.A_graph
        D_graph = self.D_graph
        D_axes = self.D_axes

        rect = Rectangle(
            height = 2.5*FRAME_Y_RADIUS,
            width = MED_SMALL_BUFF,
            stroke_width = 0,
            fill_color = YELLOW,
            fill_opacity = 0.4
        )

        self.play(
            ReplacementTransform(A_graph.copy(), thin_sum_graph),
            ReplacementTransform(D_graph.copy(), thin_sum_graph),
            # FadeOut(self.lines_to_fade)
        )
        self.play(
            self.get_graph_line_animation(self.A_axes, self.A_graph),
            self.get_graph_line_animation(self.D_axes, self.D_graph),
            self.get_graph_line_animation(axes, sum_graph.deepcopy()),
            ShowCreation(sum_graph),
            run_time = 15,
            rate_func=linear
        )
        self.remove(thin_sum_graph)
        self.wait()
        for x in 2.85, 3.57:
            rect.move_to(D_axes.coords_to_point(x, 0))
            self.play(GrowFromPoint(rect, rect.get_top()))
            self.wait()
            self.play(FadeOut(rect))

        self.sum_graph = sum_graph

    def add_more_notes(self):
        axes = self.axes

        A_group = VGroup(self.A_axes, self.A_graph, self.A_label)
        D_group = VGroup(self.D_axes, self.D_graph, self.D_label)
        squish_group = VGroup(A_group, D_group)
        squish_group.generate_target()
        squish_group.target.stretch(0.5, 1)
        squish_group.target.next_to(axes, DOWN, buff = -SMALL_BUFF)
        for group in squish_group.target:
            label = group[-1]
            bottom = label.get_bottom()
            label.stretch_in_place(0.5, 0)
            label.move_to(bottom, DOWN)

        self.play(
            MoveToTarget(squish_group),
            FadeOut(self.lines_to_fade),
        )

        F_axes = self.D_axes.deepcopy()
        C_axes = self.A_axes.deepcopy()
        VGroup(F_axes, C_axes).next_to(squish_group, DOWN)
        F_graph = self.get_wave_graph(self.A_frequency*4.0/5, F_axes)
        F_graph.set_color(self.F_color)
        C_graph = self.get_wave_graph(self.A_frequency*6.0/5, C_axes)
        C_graph.set_color(self.C_color)

        F_label = TextMobject("F349")
        C_label = TextMobject("C523")
        for label, graph in (F_label, F_graph), (C_label, C_graph):
            label.scale(0.5)
            label.set_color(graph.get_stroke_color())
            label.next_to(graph, UP, SMALL_BUFF)

        graphs = VGroup(self.A_graph, self.D_graph, F_graph, C_graph)
        def new_sum_func(x):
            result = sum([
                graph.underlying_function(x) - self.equilibrium_height
                for graph in graphs
            ])
            result *= 0.5
            return result + self.equilibrium_height
        new_sum_graph = self.axes.get_graph(
            new_sum_func, 
            num_graph_points = 200
        )
        new_sum_graph.set_color(BLUE_C)
        thin_new_sum_graph = new_sum_graph.copy().fade()

        self.play(*it.chain(
            list(map(ShowCreation, [F_axes, C_axes, F_graph, C_graph])),
            list(map(Write, [F_label, C_label])),
            list(map(FadeOut, [self.sum_graph]))
        ))
        self.play(ReplacementTransform(
            graphs.copy(), thin_new_sum_graph
        ))
        kwargs = {"rate_func" : None, "run_time" : 10}
        self.play(ShowCreation(new_sum_graph.copy(), **kwargs), *[
            self.get_graph_line_animation(curr_axes, graph, **kwargs)
            for curr_axes, graph in [
                (self.A_axes, self.A_graph),
                (self.D_axes, self.D_graph),
                (F_axes, F_graph),
                (C_axes, C_graph),
                (axes, new_sum_graph),
            ]
        ])
        self.wait()

    ####

    def broadcast(self, *added_anims, **kwargs):
        self.play(self.get_broadcast_animation(**kwargs), *added_anims)

    def get_broadcast_animation(self, **kwargs):
        kwargs["run_time"] = kwargs.get("run_time", 5)
        kwargs["n_circles"] = kwargs.get("n_circles", 10)
        return Broadcast(self.speaker[1], **kwargs)

    def get_wave_graph(self, frequency, axes):
        tail_len = 3.0
        x_min, x_max = axes.x_min, axes.x_max
        def func(x):
            value = 0.7*np.cos(2*np.pi*frequency*x)
            if x - x_min < tail_len:
                value *= smooth((x-x_min)/tail_len)
            if x_max - x < tail_len:
                value *= smooth((x_max - x )/tail_len)
            return value + self.equilibrium_height
        ngp = 2*(x_max - x_min)*frequency + 1
        graph = axes.get_graph(func, num_graph_points = int(ngp))
        return graph

    def get_A_graph_v_line(self, x):
        return self.get_graph_v_line(x, self.A_axes, self.A_graph)

    def get_D_graph_v_line(self, x):
        return self.get_graph_v_line(x, self.D_axes, self.D_graph)

    def get_graph_v_line(self, x, axes, graph):
        result = Line(
            axes.coords_to_point(x, self.equilibrium_height),
            # axes.coords_to_point(x, graph.underlying_function(x)),
            graph.point_from_proportion(float(x)/axes.x_max),
            color = WHITE,
            buff = 0,
        )
        return result

    def stack_v_lines(self, x, lines):
        point = self.axes.coords_to_point(x, self.equilibrium_height)
        A_line, D_line = lines
        A_line.shift(point - A_line.get_start())
        D_line.shift(A_line.get_end()-D_line.get_start())
        A_line.set_color(self.A_color)
        D_line.set_color(self.D_color)
        return lines

    def create_pi_creature(self):
        return Randolph().to_corner(DOWN+LEFT)

    def get_graph_line_animation(self, axes, graph, **kwargs):
        line = self.get_graph_v_line(0, axes, graph)
        x_max = axes.x_max
        def update_line(line, alpha):
            x = alpha*x_max
            #Transform(line, self.get_graph_v_line(x, axes, graph)).update(1)
            return line

        return UpdateFromAlphaFunc(line, update_line, **kwargs)
