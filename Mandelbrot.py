import pygame
import numpy as np
import moderngl
import sys

WIDTH, HEIGHT = 1280, 720
MAX_ITER = 200

VERTEX_SHADER = '''
#version 330
in vec2 in_pos;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
'''

FRAGMENT_SHADER = '''
#version 400

uniform dvec2 u_center;
uniform double u_scale;
uniform ivec2 u_resolution;
uniform int u_max_iter;
uniform int u_mode;     // 0 = Mandelbrot, 1 = Julia
uniform dvec2 u_control;

out vec4 fragColor;

int iterate(dvec2 z0, dvec2 c) {
    dvec2 z = z0;
    int i = 0;
    for (; i < u_max_iter; i++) {
        double x = z.x*z.x - z.y*z.y + c.x;
        double y = 2.0*z.x*z.y + c.y;
        z = dvec2(x, y);
        if (x*x + y*y > 4.0) break;
    }
    return i;
}

void profile_mandelbrot(dvec2 dc, dvec2 control, out dvec2 z0, out dvec2 c) {
    z0 = dvec2(0.0, 0.0);
    c  = dc;
}

void profile_julia(dvec2 dc, dvec2 control, out dvec2 z0, out dvec2 c) {
    z0 = dc;
    // base constant stored inside function, control offsets it
    c = dvec2(-0.8, 0.156) + control;
}

void main() {
    vec2 uv = vec2(gl_FragCoord.xy) / vec2(u_resolution);
    dvec2 dc = u_center + dvec2(uv.x - 0.5, uv.y - 0.5) * u_scale;

    dvec2 z0;
    dvec2 c;

    if (u_mode == 1) {
        profile_julia(dc, u_control, z0, c);
    } else {
        profile_mandelbrot(dc, u_control, z0, c);
    }

    int iter = iterate(z0, c);

    float t = pow(float(iter) / float(u_max_iter), 0.7);
    float hue = mod(0.5 * t, 1.0);
    float h = hue * 10.0;
    float s = 1.0;
    float v = (iter < u_max_iter) ? 1.0 : 0.0;
    float f = fract(h);
    float p = v * (1.0 - s);
    float q = v * (1.0 - s * f);
    float r = v * (1.0 - s * (1.0 - f));

    vec3 rgb;
    if (h < 1.0)       rgb = vec3(v, r, p);
    else if (h < 2.0)  rgb = vec3(q, v, p);
    else if (h < 3.0)  rgb = vec3(p, v, r);
    else if (h < 4.0)  rgb = vec3(p, q, v);
    else if (h < 5.0)  rgb = vec3(r, p, v);
    else               rgb = vec3(v, p, q);

    fragColor = vec4(rgb, 1.0);
}
'''

class FractalApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("Fractal")
        self.ctx = moderngl.create_context()
        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4')
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_pos')

        self.u_center = self.prog['u_center']
        self.u_scale = self.prog['u_scale']
        self.u_resolution = self.prog['u_resolution']
        self.u_max_iter = self.prog['u_max_iter']
        self.u_mode = self.prog['u_mode']
        self.u_control = self.prog['u_control']

        self.center = np.array([0.0, 0.0], dtype=np.float64)
        self.scale = 3.0
        self.max_iter = MAX_ITER
        self.mode = 0
        self.control = np.array([0.0, 0.0], dtype=np.float64)

        self.zooming_out = False
        self.zoomout_focus = None
        self.running = True

    def render(self):
        self.u_center.value = tuple(self.center.tolist())
        self.u_scale.value = float(self.scale)
        self.u_resolution.value = (WIDTH, HEIGHT)
        self.u_max_iter.value = int(self.max_iter)
        self.u_mode.value = int(self.mode)
        self.u_control.value = tuple(self.control.tolist())

        self.ctx.clear()
        self.vao.render(moderngl.TRIANGLE_STRIP)
        pygame.display.flip()

    def zoom_at(self, zoom_factor, mouse_pos):
        mx, my = mouse_pos
        rel_x = (mx / WIDTH  - 0.5) * self.scale
        rel_y = (my / HEIGHT - 0.5) * self.scale
        focus = self.center + np.array([rel_x, -rel_y], dtype=np.float64)
        self.scale *= zoom_factor
        self.center = focus - np.array([rel_x, -rel_y], dtype=np.float64) * zoom_factor

    def reset_view(self):
        self.center[:] = [0.0, 0.0]
        self.scale = 3.0
        self.control[:] = [0.0, 0.0]
        self.max_iter = MAX_ITER

    def run(self):
        clock = pygame.time.Clock()
        while self.running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.running = False
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    if e.button == 1:
                        self.zoom_at(0.9, e.pos)
                    elif e.button == 3:
                        if not self.zooming_out:
                            self.zoomout_focus = e.pos
                            self.zooming_out = True
                        else:
                            self.zoomout_focus = None
                            self.zooming_out = False
                    elif e.button == 5:
                        self.zoom_at(1.1, e.pos)
                    elif e.button == 4:
                        self.zoom_at(0.9, e.pos)
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_r:
                        self.reset_view()
                    if e.key == pygame.K_SPACE:
                        self.mode = (self.mode + 1) % 2
                        print("mode", self.mode)
                    if e.key == pygame.K_ESCAPE:
                        self.running = False

            keys = pygame.key.get_pressed()
            move = 0.01 * self.scale
            ctrl_delta = 0.005 * max(1.0, self.scale / 10.0)
            if keys[pygame.K_a]:
                self.center[0] -= move
            if keys[pygame.K_d]:
                self.center[0] += move
            if keys[pygame.K_w]:
                self.center[1] += move
            if keys[pygame.K_s]:
                self.center[1] -= move

            if keys[pygame.K_LEFT]:
                self.control[0] -= ctrl_delta
                print(f"control {self.control[0]:.7f}, {self.control[1]:.7f}")
            if keys[pygame.K_RIGHT]:
                self.control[0] += ctrl_delta
                print(f"control {self.control[0]:.7f}, {self.control[1]:.7f}")
            if keys[pygame.K_DOWN]:
                self.control[1] -= ctrl_delta
                print(f"control {self.control[0]:.7f}, {self.control[1]:.7f}")
            if keys[pygame.K_UP]:
                self.control[1] += ctrl_delta
                print(f"control {self.control[0]:.7f}, {self.control[1]:.7f}")

            if keys[pygame.K_e]:
                self.max_iter += 1
                print("iter", self.max_iter)
            if keys[pygame.K_q]:
                self.max_iter = max(1, self.max_iter - 1)
                print("iter", self.max_iter)

            if self.zooming_out:
                if self.scale < 1e6:
                    self.zoom_at(1.02, self.zoomout_focus)
                else:
                    self.zooming_out = False

            self.render()
            clock.tick(60)

        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    FractalApp().run()
